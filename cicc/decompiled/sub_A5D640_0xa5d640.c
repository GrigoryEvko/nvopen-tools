// Function: sub_A5D640
// Address: 0xa5d640
//
__int64 __fastcall sub_A5D640(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  unsigned int v12; // eax
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  _BYTE *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  _BYTE *v20; // rax
  unsigned __int8 v21; // al
  __int64 v22; // rdx
  unsigned __int8 v23; // al
  __int64 v24; // rdx
  unsigned __int8 v25; // al
  __int64 v26; // rdx
  unsigned __int8 v27; // al
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 *v31; // rdx
  unsigned int v32; // eax
  __int64 v33; // rcx
  unsigned __int8 v34; // al
  __int64 v35; // rdx
  unsigned __int8 v36; // al
  __int64 v37; // r13
  __int64 v38; // rcx
  _BYTE *v40; // rax
  __int64 v41; // [rsp+0h] [rbp-40h] BYREF
  char v42; // [rsp+8h] [rbp-38h]
  char *v43; // [rsp+10h] [rbp-30h]
  __int64 v44; // [rsp+18h] [rbp-28h]

  v4 = a2 - 16;
  sub_904010(a1, "!DICompositeType(");
  v44 = a3;
  v41 = a1;
  v43 = ", ";
  v42 = 1;
  sub_A53560(&v41, a2);
  v5 = sub_A547D0(a2, 2);
  sub_A53660(&v41, "name", 4u, v5, v6, 1);
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(_QWORD *)(a2 - 32);
  else
    v8 = v4 - 8LL * ((v7 >> 2) & 0xF);
  sub_A5CC00((__int64)&v41, "scope", 5u, *(_QWORD *)(v8 + 8), 1);
  v9 = a2;
  if ( *(_BYTE *)a2 != 16 )
    v9 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v41, "file", 4u, v9, 1);
  sub_A537C0((__int64)&v41, "line", 4u, *(_DWORD *)(a2 + 16), 1);
  v10 = *(_BYTE *)(a2 - 16);
  if ( (v10 & 2) != 0 )
    v11 = *(_QWORD *)(a2 - 32);
  else
    v11 = v4 - 8LL * ((v10 >> 2) & 0xF);
  sub_A5CC00((__int64)&v41, "baseType", 8u, *(_QWORD *)(v11 + 24), 1);
  sub_A539C0((__int64)&v41, "size", 4u, *(_QWORD *)(a2 + 24));
  v12 = sub_AF18D0(a2);
  sub_A537C0((__int64)&v41, "align", 5u, v12, 1);
  sub_A539C0((__int64)&v41, "offset", 6u, *(_QWORD *)(a2 + 32));
  sub_A537C0((__int64)&v41, "num_extra_inhabitants", 0x15u, *(_DWORD *)(a2 + 40), 1);
  sub_A53C60(&v41, "flags", 5u, *(_DWORD *)(a2 + 20));
  v13 = *(_BYTE *)(a2 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(a2 - 32);
  else
    v14 = v4 - 8LL * ((v13 >> 2) & 0xF);
  sub_A5CC00((__int64)&v41, "elements", 8u, *(_QWORD *)(v14 + 32), 1);
  sub_A53AC0(&v41, "runtimeLang", 0xBu, *(_DWORD *)(a2 + 44), (__int64 (__fastcall *)(_QWORD))sub_E0A700, 1);
  v15 = *(_BYTE *)(a2 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(a2 - 32);
  else
    v16 = v4 - 8LL * ((v15 >> 2) & 0xF);
  sub_A5CC00((__int64)&v41, "vtableHolder", 0xCu, *(_QWORD *)(v16 + 40), 1);
  v17 = sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v41, "templateParams", 0xEu, *((_QWORD *)v17 + 6), 1);
  v18 = sub_A547D0(a2, 7);
  sub_A53660(&v41, "identifier", 0xAu, v18, v19, 1);
  v20 = sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v41, "discriminator", 0xDu, *((_QWORD *)v20 + 8), 1);
  v21 = *(_BYTE *)(a2 - 16);
  if ( (v21 & 2) != 0 )
    v22 = *(_QWORD *)(a2 - 32);
  else
    v22 = v4 - 8LL * ((v21 >> 2) & 0xF);
  sub_A5CC00((__int64)&v41, "dataLocation", 0xCu, *(_QWORD *)(v22 + 72), 1);
  v23 = *(_BYTE *)(a2 - 16);
  if ( (v23 & 2) != 0 )
    v24 = *(_QWORD *)(a2 - 32);
  else
    v24 = v4 - 8LL * ((v23 >> 2) & 0xF);
  sub_A5CC00((__int64)&v41, "associated", 0xAu, *(_QWORD *)(v24 + 80), 1);
  v25 = *(_BYTE *)(a2 - 16);
  if ( (v25 & 2) != 0 )
    v26 = *(_QWORD *)(a2 - 32);
  else
    v26 = v4 - 8LL * ((v25 >> 2) & 0xF);
  sub_A5CC00((__int64)&v41, "allocated", 9u, *(_QWORD *)(v26 + 88), 1);
  v27 = *(_BYTE *)(a2 - 16);
  if ( (v27 & 2) != 0 )
    v28 = *(_QWORD *)(a2 - 32);
  else
    v28 = v4 - 8LL * ((v27 >> 2) & 0xF);
  v29 = *(_QWORD *)(v28 + 96);
  if ( v29 && *(_BYTE *)v29 == 1 && (v30 = *(_QWORD *)(v29 + 136)) != 0 && *(_BYTE *)v30 == 17 )
  {
    v31 = *(__int64 **)(v30 + 24);
    v32 = *(_DWORD *)(v30 + 32);
    if ( v32 > 0x40 )
    {
      v33 = *v31;
    }
    else
    {
      v33 = 0;
      if ( v32 )
        v33 = (__int64)((_QWORD)v31 << (64 - (unsigned __int8)v32)) >> (64 - (unsigned __int8)v32);
    }
    sub_A538D0((__int64)&v41, "rank", 4u, v33);
    v34 = *(_BYTE *)(a2 - 16);
    if ( (v34 & 2) != 0 )
      goto LABEL_27;
  }
  else
  {
    v40 = sub_A17150((_BYTE *)(a2 - 16));
    sub_A5CC00((__int64)&v41, "rank", 4u, *((_QWORD *)v40 + 12), 1);
    v34 = *(_BYTE *)(a2 - 16);
    if ( (v34 & 2) != 0 )
    {
LABEL_27:
      v35 = *(_QWORD *)(a2 - 32);
      goto LABEL_28;
    }
  }
  v35 = v4 - 8LL * ((v34 >> 2) & 0xF);
LABEL_28:
  sub_A5CC00((__int64)&v41, "annotations", 0xBu, *(_QWORD *)(v35 + 104), 1);
  v36 = *(_BYTE *)(a2 - 16);
  if ( (v36 & 2) != 0 )
    v37 = *(_QWORD *)(a2 - 32);
  else
    v37 = v4 - 8LL * ((v36 >> 2) & 0xF);
  v38 = *(_QWORD *)(v37 + 112);
  if ( v38 )
    sub_A5CC00((__int64)&v41, "specification", 0xDu, v38, 1);
  if ( *(_BYTE *)(a2 + 52) )
    sub_A53AC0(&v41, "enumKind", 8u, *(_DWORD *)(a2 + 48), (__int64 (__fastcall *)(_QWORD))sub_E0A630, 0);
  return sub_904010(a1, ")");
}
