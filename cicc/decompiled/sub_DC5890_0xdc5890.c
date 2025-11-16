// Function: sub_DC5890
// Address: 0xdc5890
//
_QWORD *__fastcall sub_DC5890(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 v6; // ax
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 v9; // rax
  _QWORD *v10; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // rbx
  _QWORD *v14; // rax
  __int16 v15; // dx
  _QWORD *v16; // r9
  _QWORD *v17; // r15
  _QWORD *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-78h]
  _BYTE *v24; // [rsp+20h] [rbp-60h] BYREF
  __int64 v25; // [rsp+28h] [rbp-58h]
  _BYTE v26[80]; // [rsp+30h] [rbp-50h] BYREF

  while ( 1 )
  {
    a3 = sub_D97090(a1, a3);
    v6 = *(_WORD *)(a2 + 24);
    if ( !v6 )
      break;
    if ( v6 != 2 )
      goto LABEL_5;
    a2 = *(_QWORD *)(a2 + 32);
    v12 = sub_D95540(a2);
    v13 = sub_D97050(a1, v12);
    if ( v13 >= sub_D97050(a1, a3) )
      return sub_DC5820(a1, a2, a3);
  }
  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_DWORD *)(v7 + 32);
  v9 = *(_QWORD *)(v7 + 24);
  if ( v8 > 0x40 )
    v9 = *(_QWORD *)(v9 + 8LL * ((v8 - 1) >> 6));
  if ( (v9 & (1LL << ((unsigned __int8)v8 - 1))) != 0 )
    return sub_DC5000(a1, a2, a3, 0);
LABEL_5:
  v10 = sub_DC2B70(a1, a2, a3, 0);
  if ( *((_WORD *)v10 + 12) == 3 )
  {
    v14 = sub_DC5000(a1, a2, a3, 0);
    if ( *((_WORD *)v14 + 12) == 4 )
    {
      v15 = *(_WORD *)(a2 + 24);
      if ( v15 == 8 )
      {
        v16 = *(_QWORD **)(a2 + 32);
        v24 = v26;
        v17 = v16;
        v25 = 0x400000000LL;
        v18 = &v16[*(_QWORD *)(a2 + 40)];
        if ( v18 != v16 )
        {
          do
          {
            v19 = sub_DC5890(a1, *v17, a3);
            v22 = (unsigned int)v25;
            if ( (unsigned __int64)(unsigned int)v25 + 1 > HIDWORD(v25) )
            {
              v23 = v19;
              sub_C8D5F0((__int64)&v24, v26, (unsigned int)v25 + 1LL, 8u, v20, v21);
              v22 = (unsigned int)v25;
              v19 = v23;
            }
            ++v17;
            *(_QWORD *)&v24[8 * v22] = v19;
            LODWORD(v25) = v25 + 1;
          }
          while ( v18 != v17 );
        }
        v10 = sub_DBFF60(a1, (unsigned int *)&v24, *(_QWORD *)(a2 + 48), 1u);
        if ( v24 != v26 )
          _libc_free(v24, &v24);
      }
      else if ( v15 == 10 )
      {
        return v14;
      }
    }
    else
    {
      return v14;
    }
  }
  return v10;
}
