// Function: sub_1A21C80
// Address: 0x1a21c80
//
unsigned __int64 __fastcall sub_1A21C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // r12
  unsigned __int64 result; // rax
  bool v9; // cc
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // r11
  __int64 *v13; // rbx
  __int64 v14; // r10
  __int64 *v15; // r12
  __int64 v16; // rsi
  unsigned __int64 v17; // r14
  _QWORD *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // r8d
  int v23; // r9d
  unsigned int v24; // ebx
  _QWORD *v25; // rax
  char v26; // al
  __int64 v27; // rsi
  unsigned __int64 v28; // rax
  __int64 v30; // [rsp+18h] [rbp-98h]
  char v31; // [rsp+27h] [rbp-89h]
  unsigned __int64 v32; // [rsp+30h] [rbp-80h]
  __int64 v33; // [rsp+38h] [rbp-78h]
  _QWORD *v34; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-68h]
  __int64 v36[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v37[2]; // [rsp+60h] [rbp-50h] BYREF
  __int64 v38[8]; // [rsp+70h] [rbp-40h] BYREF

  v7 = a2;
  if ( !*(_QWORD *)(a2 + 8) )
    return sub_1A21B40(a1, a2, a3, a4, a5, a6);
  if ( byte_4FB3F40 )
  {
    result = sub_15FA300(a2);
    if ( !(_BYTE)result )
      goto LABEL_10;
    v35 = *(_DWORD *)(a1 + 360);
    if ( v35 > 0x40 )
      sub_16A4FD0((__int64)&v34, (const void **)(a1 + 352));
    else
      v34 = *(_QWORD **)(a1 + 352);
    v11 = sub_15F2050(a2);
    v30 = sub_1632FA0(v11);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v12 = *(_QWORD *)(a2 - 8);
    else
      v12 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    v13 = (__int64 *)(v12 + 24);
    v33 = a2;
    v14 = sub_16348C0(a2) | 4;
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v33 = *(_QWORD *)(a2 - 8) + 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
    if ( v13 == (__int64 *)v33 )
    {
LABEL_38:
      result = sub_135E100((__int64 *)&v34);
LABEL_10:
      if ( !*(_QWORD *)(v7 + 8) )
        return result;
      goto LABEL_3;
    }
    v15 = v13;
    while ( 1 )
    {
      v16 = *v15;
      if ( *(_BYTE *)(*v15 + 16) != 13 )
      {
LABEL_37:
        v7 = a2;
        goto LABEL_38;
      }
      v17 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      v32 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      v31 = (v14 >> 2) & 1;
      if ( ((v14 >> 2) & 1) == 0 )
        break;
      sub_16A5D70((__int64)v36, (__int64 *)(v16 + 24), *(_DWORD *)(a1 + 360));
      v27 = v17;
      if ( !v17 )
        goto LABEL_48;
LABEL_46:
      v28 = sub_12BE0A0(v30, v27);
      sub_135E0D0((__int64)v37, *(_DWORD *)(a1 + 360), v28, 0);
      sub_16A7B50((__int64)v38, (__int64)v36, v37);
      sub_16A7200((__int64)&v34, v38);
      sub_135E100(v38);
      sub_135E100(v37);
      sub_135E100(v36);
LABEL_26:
      v24 = v35;
      if ( v35 > 0x40 )
      {
        if ( v24 - (unsigned int)sub_16A57B0((__int64)&v34) > 0x40 )
        {
LABEL_29:
          sub_1A21B40(a1, a2, v20, v21, v22, v23);
          return sub_135E100((__int64 *)&v34);
        }
        v25 = (_QWORD *)*v34;
      }
      else
      {
        v25 = v34;
      }
      if ( *(_QWORD *)(a1 + 368) < (unsigned __int64)v25 )
        goto LABEL_29;
      if ( !v31 || !v17 )
        v32 = sub_1643D30(v17, *v15);
      v26 = *(_BYTE *)(v32 + 8);
      if ( ((v26 - 14) & 0xFD) != 0 )
      {
        v14 = 0;
        if ( v26 == 13 )
          v14 = v32;
      }
      else
      {
        v14 = *(_QWORD *)(v32 + 24) | 4LL;
      }
      v15 += 3;
      if ( (__int64 *)v33 == v15 )
        goto LABEL_37;
    }
    if ( v17 )
    {
      v18 = *(_QWORD **)(v16 + 24);
      if ( *(_DWORD *)(v16 + 32) > 0x40u )
        v18 = (_QWORD *)*v18;
      v19 = sub_15A9930(v30, v14 & 0xFFFFFFFFFFFFFFF8LL);
      sub_135E0D0((__int64)v38, *(_DWORD *)(a1 + 360), *(_QWORD *)(v19 + 8LL * (unsigned int)v18 + 16), 0);
      sub_16A7200((__int64)&v34, v38);
      sub_135E100(v38);
      goto LABEL_26;
    }
    sub_16A5D70((__int64)v36, (__int64 *)(v16 + 24), *(_DWORD *)(a1 + 360));
LABEL_48:
    v27 = sub_1643D30(0, *v15);
    goto LABEL_46;
  }
LABEL_3:
  if ( !(unsigned __int8)sub_386E8D0(a1, v7) )
  {
    v9 = *(_DWORD *)(a1 + 360) <= 0x40u;
    *(_BYTE *)(a1 + 344) = 0;
    if ( !v9 )
    {
      v10 = *(_QWORD *)(a1 + 352);
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
    *(_QWORD *)(a1 + 352) = 0;
    *(_DWORD *)(a1 + 360) = 1;
  }
  return sub_386EA80(a1, v7);
}
