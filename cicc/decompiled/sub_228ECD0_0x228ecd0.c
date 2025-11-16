// Function: sub_228ECD0
// Address: 0x228ecd0
//
__int64 __fastcall sub_228ECD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char *a5,
        int a6,
        __int64 a7,
        __int64 a8)
{
  __int64 *v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // r15
  char v14; // al
  __int64 *v15; // rdx
  __int64 *v16; // rax
  unsigned int v17; // eax
  unsigned int v18; // r10d
  __int64 v19; // r15
  __int64 *v21; // rax
  char v22; // r14
  char v23; // r12
  char v24; // al
  char v25; // si
  char v26; // al
  __int64 v27; // rsi
  __int64 v28; // rsi
  _QWORD *v29; // rax
  _QWORD *v30; // rax
  bool v31; // r10
  __int64 v32; // rax
  unsigned int v33; // [rsp+Ch] [rbp-84h]
  __int64 *v34; // [rsp+10h] [rbp-80h]
  char v35; // [rsp+10h] [rbp-80h]
  unsigned int v36; // [rsp+18h] [rbp-78h]
  __int64 *v37; // [rsp+18h] [rbp-78h]
  char v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  _QWORD *v40; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v41; // [rsp+28h] [rbp-68h]
  __int64 v42; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v43; // [rsp+38h] [rbp-58h]
  _QWORD *v44; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v45; // [rsp+48h] [rbp-48h]
  _QWORD *v46; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v47; // [rsp+58h] [rbp-38h]

  v36 = a6 - 1;
  v10 = sub_DCC810(*(__int64 **)(a1 + 8), a3, a4, 0, 0);
  v11 = sub_D95540((__int64)v10);
  v12 = sub_228E360(a1, a5, v11);
  if ( !v12 )
    goto LABEL_52;
  v13 = (__int64)v12;
  v34 = v10;
  if ( !(unsigned __int8)sub_DBED40(*(_QWORD *)(a1 + 8), (__int64)v10) )
    v34 = sub_DCAF50(*(__int64 **)(a1 + 8), (__int64)v10, 0);
  v14 = sub_DBED40(*(_QWORD *)(a1 + 8), a2);
  v15 = (__int64 *)a2;
  if ( !v14 )
    v15 = sub_DCAF50(*(__int64 **)(a1 + 8), a2, 0);
  v16 = sub_DCA690(*(__int64 **)(a1 + 8), v13, (__int64)v15, 0, 0);
  LOBYTE(v17) = sub_228DFC0(a1, 0x26u, (__int64)v34, (__int64)v16);
  v18 = v17;
  if ( !(_BYTE)v17 )
  {
LABEL_52:
    if ( *((_WORD *)v10 + 12) || *(_WORD *)(a2 + 24) )
    {
      v19 = 16LL * v36;
      if ( sub_D968A0((__int64)v10) )
      {
        *(_QWORD *)(*(_QWORD *)(a7 + 48) + v19 + 8) = v10;
        sub_228CE70(a8, (__int64)v10, (__int64)a5);
        v18 = 0;
        *(_BYTE *)(*(_QWORD *)(a7 + 48) + 16LL * v36) &= 0xFAu;
      }
      else
      {
        if ( sub_D96900(a2) )
        {
          *(_QWORD *)(*(_QWORD *)(a7 + 48) + v19 + 8) = v10;
          sub_228CE70(a8, (__int64)v10, (__int64)a5);
        }
        else
        {
          *(_BYTE *)(a7 + 43) = 0;
          v37 = sub_DCAF50(*(__int64 **)(a1 + 8), (__int64)v10, 0);
          v21 = sub_DCAF50(*(__int64 **)(a1 + 8), a2, 0);
          sub_228CE50(a8, a2, (__int64)v21, (__int64)v37, (__int64)a5);
        }
        v35 = sub_DBE090(*(_QWORD *)(a1 + 8), (__int64)v10);
        v22 = sub_DBEC80(*(_QWORD *)(a1 + 8), (__int64)v10);
        v38 = sub_DBED40(*(_QWORD *)(a1 + 8), (__int64)v10);
        v23 = sub_DBEC80(*(_QWORD *)(a1 + 8), a2);
        v24 = sub_DBED40(*(_QWORD *)(a1 + 8), a2);
        v18 = 0;
        v25 = v24;
        if ( v22 || (v26 = 1, v23) )
          v26 = (v25 | v38) ^ 1;
        if ( !v35 )
          v26 |= 2u;
        if ( !v38 && !v23 || !v22 && !v25 )
          v26 |= 4u;
        *(_BYTE *)(*(_QWORD *)(a7 + 48) + v19) = *(_BYTE *)(*(_QWORD *)(a7 + 48) + v19) & 7 & v26
                                               | *(_BYTE *)(*(_QWORD *)(a7 + 48) + v19) & 0xF8;
      }
      return v18;
    }
    v27 = v10[4];
    v41 = *(_DWORD *)(v27 + 32);
    if ( v41 > 0x40 )
      sub_C43780((__int64)&v40, (const void **)(v27 + 24));
    else
      v40 = *(_QWORD **)(v27 + 24);
    v28 = *(_QWORD *)(a2 + 32);
    v43 = *(_DWORD *)(v28 + 32);
    if ( v43 > 0x40 )
      sub_C43780((__int64)&v42, (const void **)(v28 + 24));
    else
      v42 = *(_QWORD *)(v28 + 24);
    v45 = v41;
    if ( v41 > 0x40 )
    {
      sub_C43780((__int64)&v44, (const void **)&v40);
      v47 = v41;
      if ( v41 > 0x40 )
      {
        sub_C43780((__int64)&v46, (const void **)&v40);
LABEL_29:
        sub_C4C400((__int64)&v40, (__int64)&v42, (__int64)&v44, (__int64)&v46);
        v33 = v47;
        if ( v47 > 0x40 )
        {
          if ( v33 - (unsigned int)sub_C444A0((__int64)&v46) > 0x40 )
            goto LABEL_32;
          v29 = (_QWORD *)*v46;
        }
        else
        {
          v29 = v46;
        }
        if ( !v29 )
        {
          v39 = 16LL * v36;
          *(_QWORD *)(*(_QWORD *)(a7 + 48) + v39 + 8) = sub_DA26C0(*(__int64 **)(a1 + 8), (__int64)&v44);
          v30 = sub_DA26C0(*(__int64 **)(a1 + 8), (__int64)&v44);
          sub_228CE70(a8, (__int64)v30, (__int64)a5);
          if ( sub_AAD930((__int64)&v44, 0) )
          {
            *(_BYTE *)(*(_QWORD *)(a7 + 48) + v39) &= 0xF9u;
          }
          else
          {
            v31 = sub_986F30((__int64)&v44, 0);
            v32 = *(_QWORD *)(a7 + 48);
            if ( v31 )
              *(_BYTE *)(v32 + v39) &= 0xFCu;
            else
              *(_BYTE *)(v32 + v39) &= 0xFAu;
          }
          sub_969240((__int64 *)&v46);
          sub_969240((__int64 *)&v44);
          sub_969240(&v42);
          sub_969240((__int64 *)&v40);
          return 0;
        }
LABEL_32:
        sub_969240((__int64 *)&v46);
        sub_969240((__int64 *)&v44);
        sub_969240(&v42);
        sub_969240((__int64 *)&v40);
        return 1;
      }
    }
    else
    {
      v47 = v41;
      v44 = v40;
    }
    v46 = v40;
    goto LABEL_29;
  }
  return v18;
}
