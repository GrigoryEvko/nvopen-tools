// Function: sub_DDDF30
// Address: 0xdddf30
//
__int64 __fastcall sub_DDDF30(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 v6; // bx
  unsigned int v7; // r15d
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r13
  __int64 v13; // rax
  char *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 *v19; // r9
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rax
  _BYTE *v24; // [rsp+8h] [rbp-78h]
  __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  unsigned int v27; // [rsp+18h] [rbp-68h]
  __int64 v28; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-58h]
  unsigned __int64 v30; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-48h]
  __int64 v32; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v33; // [rsp+48h] [rbp-38h]

  v6 = *(_WORD *)(a2 + 28) & 7;
  v7 = *(_WORD *)(a2 + 28) & 7;
  if ( (*(_WORD *)(a2 + 28) & 2) != 0 || *(_QWORD *)(a2 + 40) != 2 )
    return v7;
  if ( !*(_BYTE *)(a1 + 1252) )
    goto LABEL_11;
  v10 = *(__int64 **)(a1 + 1232);
  a4 = *(unsigned int *)(a1 + 1244);
  a3 = &v10[a4];
  if ( v10 == a3 )
  {
LABEL_10:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 1240) )
    {
      v11 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a1 + 1244) = v11;
      *a3 = a2;
      ++*(_QWORD *)(a1 + 1224);
LABEL_12:
      v12 = sub_D33D80((_QWORD *)a2, a1, (__int64)a3, v11, a5);
      v13 = sub_D95540(**(_QWORD **)(a2 + 32));
      v27 = sub_D97050(a1, v13);
      v14 = *(char **)(a2 + 48);
      v26 = *(_QWORD *)(a2 + 48);
      v15 = sub_DCF3A0((__int64 *)a1, v14, 1);
      if ( !sub_D96A50(v15) || *(_BYTE *)(a1 + 16) )
        goto LABEL_37;
      v23 = *(_QWORD *)(a1 + 32);
      if ( !*(_BYTE *)(v23 + 192) )
      {
        v25 = *(_QWORD *)(a1 + 32);
        sub_CFDFC0(v25, (__int64)v14, v16, v17, v18, v19);
        v23 = v25;
      }
      if ( *(_DWORD *)(v23 + 24) )
      {
LABEL_37:
        if ( (unsigned __int8)sub_DBEDC0(a1, v12) )
        {
          v20 = sub_DBB9F0(a1, v12, 0, 0);
          sub_AB0910((__int64)&v30, v20);
          v29 = v27;
          if ( v27 > 0x40 )
            sub_C43690((__int64)&v28, 0, 0);
          else
            v28 = 0;
          if ( v31 > 0x40 )
          {
            sub_C43D10((__int64)&v30);
          }
          else
          {
            v21 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v31) & ~v30;
            if ( !v31 )
              v21 = 0;
            v30 = v21;
          }
          sub_C46250((__int64)&v30);
          sub_C45EE0((__int64)&v30, &v28);
          v22 = v31;
          v31 = 0;
          v33 = v22;
          v32 = v30;
          v24 = sub_DA26C0((__int64 *)a1, (__int64)&v32);
          if ( v33 > 0x40 && v32 )
            j_j___libc_free_0_0(v32);
          sub_969240(&v28);
          sub_969240((__int64 *)&v30);
          if ( (unsigned __int8)sub_DDDA00(a1, v26, 0x24u, a2, v24)
            || (unsigned __int8)sub_DDDEB0((__int64 *)a1, 0x24u, a2, v24) )
          {
            return v6 | 2u;
          }
        }
      }
      return v7;
    }
LABEL_11:
    sub_C8CC70(a1 + 1224, a2, (__int64)a3, a4, a5, a6);
    if ( !(_BYTE)a3 )
      return v7;
    goto LABEL_12;
  }
  while ( a2 != *v10 )
  {
    if ( a3 == ++v10 )
      goto LABEL_10;
  }
  return v7;
}
