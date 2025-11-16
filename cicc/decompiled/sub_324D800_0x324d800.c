// Function: sub_324D800
// Address: 0x324d800
//
__int64 __fastcall sub_324D800(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int8 v6; // dl
  __int64 v7; // rdi
  __int64 v8; // rax
  int v9; // ebx
  size_t v10; // rdx
  size_t v11; // r10
  size_t v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  __int64 result; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  int v21; // ebx
  unsigned __int8 v22; // al
  __int64 v23; // r15
  unsigned __int16 v24; // ax
  bool v25; // r10
  unsigned int v26; // eax
  unsigned int v27; // eax
  const void *v28; // [rsp+0h] [rbp-60h]
  size_t v29; // [rsp+8h] [rbp-58h]
  size_t v30; // [rsp+10h] [rbp-50h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 (__fastcall *v32)(__int64 *, __int64); // [rsp+18h] [rbp-48h]
  int v33; // [rsp+28h] [rbp-38h]
  unsigned int v34; // [rsp+28h] [rbp-38h]

  v3 = a3 - 16;
  v6 = *(_BYTE *)(a3 - 16);
  if ( (v6 & 2) != 0 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)(a3 - 32) + 16LL);
    if ( !v7 )
    {
      v9 = *(unsigned __int16 *)(a2 + 28);
      v12 = 0;
      v11 = 0;
      v28 = 0;
      v31 = *(_QWORD *)(a3 + 24) >> 3;
      goto LABEL_22;
    }
  }
  else
  {
    v7 = *(_QWORD *)(v3 - 8LL * ((v6 >> 2) & 0xF) + 16);
    if ( !v7 )
    {
      v9 = *(unsigned __int16 *)(a2 + 28);
      v12 = 0;
      v11 = 0;
      v28 = 0;
      v31 = *(_QWORD *)(a3 + 24) >> 3;
      goto LABEL_4;
    }
  }
  v8 = sub_B91420(v7);
  v9 = *(unsigned __int16 *)(a2 + 28);
  v28 = (const void *)v8;
  v11 = v10;
  v6 = *(_BYTE *)(a3 - 16);
  v12 = v11;
  v31 = *(_QWORD *)(a3 + 24) >> 3;
  if ( (v6 & 2) == 0 )
  {
LABEL_4:
    v13 = v3 - 8LL * ((v6 >> 2) & 0xF);
    goto LABEL_5;
  }
LABEL_22:
  v13 = *(_QWORD *)(a3 - 32);
LABEL_5:
  v14 = *(_QWORD *)(v13 + 24);
  if ( v14 )
  {
    v29 = v12;
    v30 = v11;
    sub_32495E0(a1, a2, v14, 73);
    v12 = v29;
    v11 = v30;
  }
  if ( v11 )
    sub_324AD70(a1, a2, 3, v28, v12);
  v15 = *(_BYTE *)(a3 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(a3 - 32);
  else
    v16 = v3 - 8LL * ((v15 >> 2) & 0xF);
  sub_324CC60(a1, a2, *(_QWORD *)(v16 + 40));
  if ( (_WORD)v9 != 22 )
  {
    if ( (_WORD)v9 == 15
      || v31 == 0
      || (unsigned __int16)(v9 - 16) <= 0x32u && (v18 = 0x4000000008001LL, _bittest64(&v18, (unsigned int)(v9 - 16))) )
    {
      if ( (_WORD)v9 == 31 )
      {
        v32 = *(__int64 (__fastcall **)(__int64 *, __int64))(*a1 + 40);
        v19 = sub_AF2CE0(a3);
        v20 = v32(a1, v19);
        sub_32494F0(a1, a2, 29, v20);
      }
      goto LABEL_15;
    }
LABEL_14:
    BYTE2(v33) = 0;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 11, v33, v31);
    goto LABEL_15;
  }
  v24 = sub_3220AA0(a1[26]);
  v25 = v31 != 0;
  if ( v24 > 4u )
  {
    v26 = sub_AF18D0(a3);
    v25 = v31 != 0;
    v27 = v26 >> 3;
    if ( v27 )
    {
      v33 = 65551;
      sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 136, 65551, v27 & 0x1FFFFFFF);
      v25 = v31 != 0;
    }
  }
  if ( v25 )
    goto LABEL_14;
LABEL_15:
  sub_3249F00(a1, a2, *(_DWORD *)(a3 + 20));
  if ( (*(_BYTE *)(a3 + 20) & 4) == 0 )
    sub_3249E10(a1, a2, a3);
  if ( *(_BYTE *)(a3 + 48) )
  {
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 51, 65542, *(unsigned int *)(a3 + 44));
    if ( (_WORD)v9 != 67 )
      goto LABEL_19;
  }
  else if ( (_WORD)v9 != 67 )
  {
    goto LABEL_19;
  }
  v22 = *(_BYTE *)(a3 - 16);
  if ( (v22 & 2) != 0 )
    v23 = *(_QWORD *)(a3 - 32);
  else
    v23 = v3 - 8LL * ((v22 >> 2) & 0xF);
  sub_324D230(a1, a2, *(_QWORD *)(v23 + 32));
LABEL_19:
  result = sub_AF2E40(a3);
  v34 = result;
  if ( BYTE4(result) )
  {
    v21 = result;
    sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 15876, 65547, result & 0xF);
    if ( (v34 & 0x10) != 0 )
      sub_3249FA0(a1, a2, 15877);
    result = sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), 15878, 65541, (unsigned __int16)(v34 >> 5));
    if ( (v34 & 0x200000) != 0 )
      result = sub_3249FA0(a1, a2, 15880);
    if ( (v21 & 0x400000) != 0 )
      return sub_3249FA0(a1, a2, 15881);
  }
  return result;
}
