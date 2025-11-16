// Function: sub_EDB710
// Address: 0xedb710
//
__int64 __fastcall sub_EDB710(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int *v4; // rbx
  unsigned __int64 v5; // rsi
  int *v6; // rbx
  int v7; // r13d
  __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // r8
  _QWORD *v12; // rdx
  _QWORD *v13; // r9
  _QWORD *v14; // r8
  __int64 v16; // [rsp+0h] [rbp-60h]
  __int64 v17; // [rsp+0h] [rbp-60h]
  _QWORD *v18; // [rsp+8h] [rbp-58h]
  __int64 *v19; // [rsp+8h] [rbp-58h]
  _QWORD *v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+8h] [rbp-58h]
  _QWORD *v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v24; // [rsp+18h] [rbp-48h]
  int v25; // [rsp+20h] [rbp-40h]
  int v26; // [rsp+24h] [rbp-3Ch]
  char v27; // [rsp+28h] [rbp-38h]

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  v4 = (unsigned int *)(*(_QWORD *)a2 + 4LL * a3);
  v5 = *v4;
  v6 = (int *)(v4 + 1);
  v7 = v5;
  sub_ED87A0((__int64 *)a1, v5);
  if ( (_DWORD)v5 )
  {
    do
    {
      if ( *v6 < 0 )
        v6 += (unsigned int)-*v6;
      (*(void (__fastcall **)(__int64 *, _QWORD))(a2 + 8))(&v23, *(_QWORD *)(a2 + 16));
      v11 = *(_QWORD *)(a1 + 8);
      if ( v11 == *(_QWORD *)(a1 + 16) )
      {
        sub_EDB300((__int64 *)a1, *(__int64 **)(a1 + 8), &v23);
      }
      else
      {
        if ( v11 )
        {
          v8 = v23;
          *(_QWORD *)(v11 + 8) = 0;
          *(_QWORD *)(v11 + 16) = 0;
          *(_QWORD *)v11 = v8;
          v9 = v24;
          *(_BYTE *)(v11 + 24) = 0;
          v18 = v9;
          if ( v9 )
          {
            v16 = v11;
            v10 = (__int64 *)sub_22077B0(32);
            v11 = v16;
            if ( v10 )
            {
              v12 = v18;
              v19 = v10;
              *v10 = (__int64)(v10 + 2);
              sub_ED71E0(v10, (_BYTE *)*v12, *v12 + v12[1]);
              v11 = v16;
              v10 = v19;
            }
            v13 = *(_QWORD **)(v11 + 8);
            *(_QWORD *)(v11 + 8) = v10;
            if ( v13 )
            {
              if ( (_QWORD *)*v13 != v13 + 2 )
              {
                v17 = v11;
                v20 = v13;
                j_j___libc_free_0(*v13, v13[2] + 1LL);
                v11 = v17;
                v13 = v20;
              }
              v21 = v11;
              j_j___libc_free_0(v13, 32);
              v11 = v21;
            }
          }
          *(_DWORD *)(v11 + 16) = v25;
          *(_DWORD *)(v11 + 20) = v26;
          *(_BYTE *)(v11 + 24) = v27;
          v11 = *(_QWORD *)(a1 + 8);
        }
        *(_QWORD *)(a1 + 8) = v11 + 32;
      }
      v14 = v24;
      if ( v24 )
      {
        if ( (_QWORD *)*v24 != v24 + 2 )
        {
          v22 = v24;
          j_j___libc_free_0(*v24, v24[2] + 1LL);
          v14 = v22;
        }
        j_j___libc_free_0(v14, 32);
      }
      ++v6;
      --v7;
    }
    while ( v7 );
  }
  return a1;
}
