// Function: sub_35DF360
// Address: 0x35df360
//
__int64 __fastcall sub_35DF360(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r14
  __int64 v9; // r15
  unsigned __int8 **v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // r13d
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int8 **v16; // rax
  unsigned __int8 *v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // r8
  unsigned __int64 *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  int v23; // r14d
  unsigned int v24; // esi
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rsi
  unsigned int v31; // eax
  unsigned int v32; // [rsp+10h] [rbp-C0h]
  unsigned int v33; // [rsp+18h] [rbp-B8h]
  unsigned __int64 *v34; // [rsp+20h] [rbp-B0h]
  __int64 v35; // [rsp+28h] [rbp-A8h]
  __int64 v36; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v37; // [rsp+38h] [rbp-98h]
  __int64 *v38; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v39; // [rsp+48h] [rbp-88h]
  __int64 v40; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v41; // [rsp+58h] [rbp-78h]
  __int64 v42; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v43; // [rsp+68h] [rbp-68h]
  __int64 v44; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v45; // [rsp+78h] [rbp-58h]

  v6 = *a2;
  if ( (unsigned __int8)v6 <= 0x1Cu )
    return 1;
  v9 = a1 + 192;
  if ( *(_BYTE *)(a1 + 220) )
  {
    v10 = *(unsigned __int8 ***)(a1 + 200);
    v11 = (__int64)&v10[*(unsigned int *)(a1 + 212)];
    if ( v10 != (unsigned __int8 **)v11 )
    {
      while ( a2 != *v10 )
      {
        if ( (unsigned __int8 **)v11 == ++v10 )
          goto LABEL_11;
      }
      return 1;
    }
  }
  else
  {
    if ( sub_C8CA60(a1 + 192, (__int64)a2) )
      return 1;
    v6 = *a2;
  }
LABEL_11:
  if ( (unsigned __int8)(v6 - 49) <= 0x14u && (v11 = 1048713, _bittest64(&v11, (unsigned int)(v6 - 49)))
    || (unsigned __int8)v6 <= 0x36u && (v15 = 0x40540000000000LL, _bittest64(&v15, v6)) && !sub_B448F0((__int64)a2) )
  {
    if ( (((_BYTE)v6 - 42) & 0xFD) != 0 )
      return 0;
    v14 = *((_QWORD *)a2 + 2);
    if ( !v14 )
      return 0;
    if ( *(_QWORD *)(v14 + 8) )
      return 0;
    v35 = *(_QWORD *)(v14 + 24);
    if ( *(_BYTE *)v35 != 82 )
      return 0;
    v17 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    if ( **((_BYTE **)v17 + 4) != 17 )
      return 0;
    LOBYTE(v18) = sub_B532B0(*(_WORD *)(v35 + 2) & 0x3F);
    v12 = v18;
    if ( (_BYTE)v18 )
      return 0;
    if ( (*(_WORD *)(v35 + 2) & 0x3Fu) - 32 <= 1 )
      return 0;
    v20 = *(unsigned __int64 **)(v35 - 64);
    v34 = v20;
    if ( *(_BYTE *)v20 != 17 )
    {
      v34 = *(unsigned __int64 **)(v35 - 32);
      if ( *(_BYTE *)v34 != 17 )
        return 0;
    }
    if ( (a2[7] & 0x40) != 0 )
      v21 = *((_QWORD *)a2 - 1);
    else
      v21 = (__int64)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v22 = *(_QWORD *)(v21 + 32);
    v23 = v6 - 29;
    v24 = *(_DWORD *)(v22 + 32);
    v37 = v24;
    if ( v24 > 0x40 )
    {
      sub_C43780((__int64)&v36, (const void **)(v22 + 24));
      if ( v23 != 15 )
        goto LABEL_71;
    }
    else
    {
      v36 = *(_QWORD *)(v22 + 24);
      if ( v23 != 15 )
      {
        v25 = v24 - 1;
        v26 = 1LL << ((unsigned __int8)v24 - 1);
        goto LABEL_40;
      }
    }
    v43 = v37;
    if ( v37 > 0x40 )
      sub_C43780((__int64)&v42, (const void **)&v36);
    else
      v42 = v36;
    sub_AADAA0((__int64)&v44, (__int64)&v42, v21, (__int64)v20, v19);
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
    v36 = v44;
    v31 = v45;
    v45 = 0;
    v37 = v31;
    sub_969240(&v44);
    sub_969240(&v42);
LABEL_71:
    v24 = v37;
    v25 = v37 - 1;
    v26 = 1LL << ((unsigned __int8)v37 - 1);
    if ( v37 > 0x40 )
    {
      if ( (*(_QWORD *)(v36 + 8LL * ((unsigned int)v25 >> 6)) & v26) == 0 )
      {
        v33 = v37;
        if ( v33 != (unsigned int)sub_C444A0((__int64)&v36) )
          goto LABEL_74;
      }
      goto LABEL_41;
    }
LABEL_40:
    if ( (v26 & v36) == 0 && v36 )
    {
      if ( v24 != 64 )
      {
        v41 = v24;
        v40 = v36;
        sub_AADAA0((__int64)&v42, (__int64)&v40, v36, v25, v19);
        sub_C449B0((__int64)&v44, (const void **)&v42, 0x40u);
        sub_AADAA0((__int64)&v38, (__int64)&v44, v28, v29, (__int64)&v38);
        sub_969240(&v44);
        sub_969240(&v42);
        sub_969240(&v40);
        if ( v39 > 0x40 )
        {
          v30 = *v38;
        }
        else if ( v39 )
        {
          v30 = (__int64)((_QWORD)v38 << (64 - (unsigned __int8)v39)) >> (64 - (unsigned __int8)v39);
        }
        else
        {
          v30 = 0;
        }
        if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 8) + 1320LL))(
               *(_QWORD *)(a1 + 8),
               v30) )
        {
          sub_969240((__int64 *)&v38);
          goto LABEL_41;
        }
        sub_969240((__int64 *)&v38);
      }
LABEL_74:
      if ( v37 > 0x40 && v36 )
      {
        j_j___libc_free_0_0(v36);
        return v12;
      }
      return 0;
    }
LABEL_41:
    sub_BED950((__int64)&v44, a1 + 288, (__int64)a2);
    v11 = v37;
    v32 = v37;
    if ( v37 > 0x40 )
    {
      v11 = v32 - (unsigned int)sub_C444A0((__int64)&v36);
      if ( (unsigned int)v11 <= 0x40 )
      {
        v27 = v36;
        if ( !*(_QWORD *)v36 )
        {
LABEL_54:
          if ( v27 )
            j_j___libc_free_0_0(v27);
          goto LABEL_19;
        }
      }
      if ( (int)sub_C49970((__int64)&v36, v34 + 3) > 0 )
      {
LABEL_53:
        v27 = v36;
        goto LABEL_54;
      }
    }
    else if ( !v36 || (int)sub_C49970((__int64)&v36, v34 + 3) > 0 )
    {
      goto LABEL_19;
    }
    sub_BED950((__int64)&v44, a1 + 288, v35);
    if ( v37 <= 0x40 )
      goto LABEL_19;
    goto LABEL_53;
  }
LABEL_19:
  if ( !*(_BYTE *)(a1 + 220) )
    goto LABEL_26;
  v16 = *(unsigned __int8 ***)(a1 + 200);
  a4 = *(unsigned int *)(a1 + 212);
  v11 = (__int64)&v16[a4];
  if ( v16 != (unsigned __int8 **)v11 )
  {
    while ( a2 != *v16 )
    {
      if ( (unsigned __int8 **)v11 == ++v16 )
        goto LABEL_25;
    }
    return 1;
  }
LABEL_25:
  if ( (unsigned int)a4 < *(_DWORD *)(a1 + 208) )
  {
    *(_DWORD *)(a1 + 212) = a4 + 1;
    *(_QWORD *)v11 = a2;
    ++*(_QWORD *)(a1 + 192);
  }
  else
  {
LABEL_26:
    sub_C8CC70(v9, (__int64)a2, v11, a4, a5, a6);
  }
  return 1;
}
