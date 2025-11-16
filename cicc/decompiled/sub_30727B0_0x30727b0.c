// Function: sub_30727B0
// Address: 0x30727b0
//
unsigned __int64 __fastcall sub_30727B0(__int64 a1, __int64 a2, char a3, char a4)
{
  int v5; // r14d
  unsigned __int64 v6; // rax
  unsigned int v7; // edx
  unsigned __int64 v8; // rbx
  unsigned int i; // r12d
  unsigned __int64 v10; // rax
  __int64 *v11; // r11
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // eax
  bool v15; // of
  __int64 *v16; // r11
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned int v19; // eax
  __int64 *v20; // [rsp+0h] [rbp-60h]
  __int64 *v21; // [rsp+0h] [rbp-60h]
  unsigned int v24; // [rsp+1Ch] [rbp-44h]
  unsigned __int64 v25; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-38h]

  if ( *(_BYTE *)(a2 + 8) == 18 )
    return 0;
  v5 = *(_DWORD *)(a2 + 32);
  v26 = v5;
  if ( (unsigned int)v5 <= 0x40 )
  {
    v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
    if ( !v5 )
      v6 = 0;
    v7 = v5;
    v25 = v6;
    goto LABEL_7;
  }
  sub_C43690((__int64)&v25, -1, 1);
  if ( *(_BYTE *)(a2 + 8) != 18 )
  {
    v5 = *(_DWORD *)(a2 + 32);
    v7 = v26;
LABEL_7:
    v8 = 0;
    if ( v5 > 0 )
    {
      for ( i = 0; i != v5; ++i )
      {
        v10 = v25;
        if ( v7 > 0x40 )
          v10 = *(_QWORD *)(v25 + 8LL * (i >> 6));
        if ( (v10 & (1LL << i)) != 0 )
        {
          if ( a3 )
          {
            v11 = (__int64 *)a2;
            if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
              v11 = **(__int64 ***)(a2 + 16);
            v20 = v11;
            v12 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v11, 0);
            BYTE2(v24) = 0;
            v14 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 24) + 736LL))(
                    *(_QWORD *)(a1 + 24),
                    *v20,
                    v12,
                    v13,
                    v24);
            v15 = __OFADD__(v14, v8);
            v8 += v14;
            if ( v15 )
            {
              v8 = 0x8000000000000000LL;
              if ( v14 )
                v8 = 0x7FFFFFFFFFFFFFFFLL;
            }
          }
          if ( a4 )
          {
            v16 = (__int64 *)a2;
            if ( (unsigned int)*(unsigned __int8 *)(a2 + 8) - 17 <= 1 )
              v16 = **(__int64 ***)(a2 + 16);
            v21 = v16;
            v17 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v16, 0);
            BYTE2(v24) = 0;
            v19 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 24) + 736LL))(
                    *(_QWORD *)(a1 + 24),
                    *v21,
                    v17,
                    v18,
                    v24);
            v7 = v26;
            v15 = __OFADD__(v19, v8);
            v8 += v19;
            if ( v15 )
            {
              v8 = 0x8000000000000000LL;
              if ( v19 )
                v8 = 0x7FFFFFFFFFFFFFFFLL;
            }
          }
          else
          {
            v7 = v26;
          }
        }
      }
    }
    goto LABEL_27;
  }
  v7 = v26;
  v8 = 0;
LABEL_27:
  if ( v7 > 0x40 )
  {
    if ( v25 )
      j_j___libc_free_0_0(v25);
  }
  return v8;
}
