// Function: sub_939CF0
// Address: 0x939cf0
//
__int64 __fastcall sub_939CF0(_QWORD **a1, __int64 a2, unsigned __int8 a3, _DWORD *a4)
{
  __int64 v5; // rdx
  unsigned int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 v12; // rbx
  __int64 v13; // rax
  _BYTE *v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // r12
  __int64 v18; // rax
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v22; // [rsp+28h] [rbp-48h]
  _BYTE *v23; // [rsp+30h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 16);
  v22 = 0;
  v23 = 0;
  v6 = *(_DWORD *)(v5 + 12);
  v21 = 0;
  if ( v6 == 2 )
    sub_91B8A0("indirect return not supported!", a4, 1);
  if ( v6 > 2 )
  {
    if ( v6 != 3 )
      sub_91B8A0("unknown ABI variant for return type!", a4, 1);
    v8 = sub_BCB120(**a1);
  }
  else
  {
    v8 = sub_91A390((__int64)a1, *(_QWORD *)(v5 + 24), 0, (__int64)a4);
  }
  v9 = *(_QWORD *)(a2 + 16);
  v10 = 5LL * *(unsigned int *)(a2 + 8) + 5;
  v11 = v9 + 40;
  v12 = v9 + 8 * v10;
  if ( v12 != v9 + 40 )
  {
    do
    {
      v15 = *(_DWORD *)(v11 + 12);
      if ( v15 == 2 )
      {
        v18 = sub_91A3A0((__int64)a1, *(_QWORD *)(v11 + 24), v10, v7);
        v13 = sub_BCE760(v18, 0);
        v14 = v22;
        v20 = v13;
        if ( v22 != v23 )
        {
LABEL_7:
          if ( v14 )
          {
            *(_QWORD *)v14 = v13;
            v14 = v22;
          }
          v22 = v14 + 8;
          goto LABEL_10;
        }
      }
      else
      {
        if ( v15 > 2 )
        {
          if ( v15 != 3 )
            sub_91B8A0("unknown ABI variant for argument!", a4, 1);
          goto LABEL_10;
        }
        v13 = sub_91A390((__int64)a1, *(_QWORD *)(v11 + 24), 0, v7);
        v14 = v22;
        v20 = v13;
        if ( v22 != v23 )
          goto LABEL_7;
      }
      sub_9183A0((__int64)&v21, v14, &v20);
LABEL_10:
      v11 += 40;
    }
    while ( v11 != v12 );
  }
  v16 = sub_BCF480(v8, v21, (__int64)&v22[-v21] >> 3, a3);
  if ( v21 )
    j_j___libc_free_0(v21, &v23[-v21]);
  return v16;
}
