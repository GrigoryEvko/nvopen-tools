// Function: sub_1BE8450
// Address: 0x1be8450
//
void __fastcall sub_1BE8450(__int64 a1, int *a2)
{
  __int64 v3; // rsi
  bool v4; // zf
  __int64 v5; // r12
  int v6; // r14d
  int j; // r13d
  __int64 v8; // r15
  __int64 v9; // rdi
  __int64 v10; // r13
  __int64 v11; // rdi
  int v12; // [rsp+8h] [rbp-58h]
  int i; // [rsp+Ch] [rbp-54h]
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+18h] [rbp-48h]
  __int64 v16; // [rsp+20h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 112);
  v14 = 0;
  v15 = 0;
  v16 = 0;
  sub_1BE7710((__int64)&v14, v3);
  if ( *(_BYTE *)(a1 + 128) )
  {
    v4 = *((_BYTE *)a2 + 16) == 0;
    *((_QWORD *)a2 + 1) = 0;
    if ( v4 )
      *((_BYTE *)a2 + 16) = 1;
    v5 = v14;
    v12 = a2[1];
    if ( !v12 )
      goto LABEL_17;
    for ( i = 0; i != v12; ++i )
    {
      v6 = *a2;
      a2[2] = i;
      if ( v6 )
      {
        for ( j = 0; j != v6; ++j )
        {
          v8 = v15;
          a2[3] = j;
          if ( v8 != v5 )
          {
            do
            {
              v9 = *(_QWORD *)(v8 - 8);
              v8 -= 8;
              (*(void (__fastcall **)(__int64, int *))(*(_QWORD *)v9 + 16LL))(v9, a2);
            }
            while ( v8 != v5 );
            v5 = v14;
          }
        }
      }
    }
    if ( *((_BYTE *)a2 + 16) )
LABEL_17:
      *((_BYTE *)a2 + 16) = 0;
  }
  else
  {
    v5 = v15;
    v10 = v14;
    if ( v14 != v15 )
    {
      do
      {
        v11 = *(_QWORD *)(v5 - 8);
        v5 -= 8;
        (*(void (__fastcall **)(__int64, int *))(*(_QWORD *)v11 + 16LL))(v11, a2);
      }
      while ( v10 != v5 );
      v5 = v14;
    }
  }
  if ( v5 )
    j_j___libc_free_0(v5, v16 - v5);
}
