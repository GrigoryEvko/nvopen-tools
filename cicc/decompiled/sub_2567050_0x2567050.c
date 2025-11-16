// Function: sub_2567050
// Address: 0x2567050
//
void __fastcall sub_2567050(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r13
  unsigned int v4; // eax
  __int64 v5; // rsi
  __int64 v6; // rdi
  unsigned int v7; // eax
  unsigned __int8 *v8; // r12
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rdi
  unsigned int v13; // eax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  __int64 v16; // rcx
  unsigned int v17; // eax
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rax
  const void **v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  __int64 v28; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-58h]
  __int64 v30[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v31[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = (_QWORD *)(a1 + 72);
  v26 = (const void **)(a1 + 136);
  v27 = a1 + 104;
  if ( sub_2566C40(a2 + 32, (__int64 *)(a1 + 72)) )
  {
    if ( *(_DWORD *)(a1 + 112) <= 0x40u && (v4 = *(_DWORD *)(a1 + 144), v4 <= 0x40) )
    {
      v21 = *(_QWORD *)(a1 + 136);
      *(_DWORD *)(a1 + 112) = v4;
      *(_QWORD *)(a1 + 104) = v21;
    }
    else
    {
      sub_C43990(v27, (__int64)v26);
    }
    v5 = a1 + 152;
    v6 = a1 + 120;
    if ( *(_DWORD *)(a1 + 128) > 0x40u || (v7 = *(_DWORD *)(a1 + 160), v7 > 0x40) )
    {
      sub_C43990(v6, v5);
      if ( *(_DWORD *)(a1 + 112) <= 0x40u )
        goto LABEL_8;
LABEL_20:
      if ( !sub_C43C50(v27, v26) )
        goto LABEL_9;
      goto LABEL_21;
    }
    v20 = *(_QWORD *)(a1 + 152);
    *(_DWORD *)(a1 + 128) = v7;
    *(_QWORD *)(a1 + 120) = v20;
  }
  else
  {
    v14 = sub_2509740(v2);
    sub_254EAA0((__int64)v30, a1, a2, v14);
    sub_254F8E0(a1 + 88, (__int64)v30);
    sub_969240(v31);
    sub_969240(v30);
    v15 = sub_2509740(v2);
    sub_254EE20((__int64)v30, a1, a2, v15);
    sub_254F8E0(a1 + 88, (__int64)v30);
    sub_969240(v31);
    sub_969240(v30);
  }
  if ( *(_DWORD *)(a1 + 112) > 0x40u )
    goto LABEL_20;
LABEL_8:
  if ( *(_QWORD *)(a1 + 104) != *(_QWORD *)(a1 + 136) )
    goto LABEL_9;
LABEL_21:
  if ( *(_DWORD *)(a1 + 128) <= 0x40u )
  {
    if ( *(_QWORD *)(a1 + 120) == *(_QWORD *)(a1 + 152) )
      return;
  }
  else if ( sub_C43C50(a1 + 120, (const void **)(a1 + 152)) )
  {
    return;
  }
LABEL_9:
  v8 = (unsigned __int8 *)sub_250D070(v2);
  LODWORD(v9) = *v8;
  if ( (_BYTE)v9 == 17 )
  {
    v29 = *((_DWORD *)v8 + 8);
    if ( v29 > 0x40 )
      sub_C43780((__int64)&v28, (const void **)v8 + 3);
    else
      v28 = *((_QWORD *)v8 + 3);
LABEL_12:
    sub_AADBC0((__int64)v30, &v28);
    sub_254F7F0(a1 + 88, (__int64)v30);
    sub_969240(v31);
    sub_969240(v30);
    sub_969240(&v28);
    if ( *(_DWORD *)(a1 + 144) <= 0x40u && (v10 = *(_DWORD *)(a1 + 112), v10 <= 0x40) )
    {
      v22 = *(_QWORD *)(a1 + 104);
      *(_DWORD *)(a1 + 144) = v10;
      *(_QWORD *)(a1 + 136) = v22;
    }
    else
    {
      sub_C43990((__int64)v26, v27);
    }
    v11 = a1 + 120;
    v12 = a1 + 152;
    if ( *(_DWORD *)(a1 + 160) > 0x40u || (v13 = *(_DWORD *)(a1 + 128), v13 > 0x40) )
    {
LABEL_17:
      sub_C43990(v12, v11);
      return;
    }
    v23 = *(_QWORD *)(a1 + 120);
    *(_DWORD *)(a1 + 160) = v13;
    *(_QWORD *)(a1 + 152) = v23;
    return;
  }
  if ( (unsigned int)(unsigned __int8)v9 - 12 <= 1 )
  {
    v29 = *(_DWORD *)(a1 + 96);
    if ( v29 > 0x40 )
      sub_C43690((__int64)&v28, 0, 0);
    else
      v28 = 0;
    goto LABEL_12;
  }
  if ( (unsigned __int8)v9 <= 0x1Cu )
    goto LABEL_54;
  if ( (unsigned __int8)(v9 - 34) > 0x33u || (v16 = 0xB3FFE03FFFF41LL, !_bittest64(&v16, (unsigned int)(v9 - 34))) )
  {
    if ( (_BYTE)v9 == 61 )
    {
      if ( (v8[7] & 0x20) == 0 )
        goto LABEL_54;
      v25 = sub_B91C10((__int64)v8, 4);
      if ( v25 )
      {
        sub_ABEA30((__int64)v30, v25);
        sub_254F8E0(a1 + 88, (__int64)v30);
        sub_969240(v31);
        sub_969240(v30);
        return;
      }
      v9 = *v8;
      if ( (unsigned __int8)v9 <= 0x1Cu )
        goto LABEL_54;
    }
    if ( (v9 & 0xFD) != 0x54 )
    {
LABEL_54:
      if ( *(_DWORD *)(a1 + 112) <= 0x40u && (v17 = *(_DWORD *)(a1 + 144), v17 <= 0x40) )
      {
        v24 = *(_QWORD *)(a1 + 136);
        *(_DWORD *)(a1 + 112) = v17;
        *(_QWORD *)(a1 + 104) = v24;
      }
      else
      {
        sub_C43990(v27, (__int64)v26);
      }
      v11 = a1 + 152;
      v12 = a1 + 120;
      if ( *(_DWORD *)(a1 + 128) > 0x40u )
        goto LABEL_17;
      v18 = *(_DWORD *)(a1 + 160);
      if ( v18 > 0x40 )
        goto LABEL_17;
      v19 = *(_QWORD *)(a1 + 152);
      *(_DWORD *)(a1 + 128) = v18;
      *(_QWORD *)(a1 + 120) = v19;
    }
  }
}
