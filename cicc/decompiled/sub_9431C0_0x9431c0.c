// Function: sub_9431C0
// Address: 0x9431c0
//
__int64 __fastcall sub_9431C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char *v6; // r13
  const char *v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // r9d
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r15
  int v15; // r9d
  char v16; // si
  int v17; // r8d
  int v18; // r11d
  int v19; // eax
  __int64 v20; // r14
  __int64 v21; // r11
  int v22; // eax
  __int64 v23; // r15
  __int64 v24; // rax
  int v26; // r11d
  int v27; // eax
  __int64 v28; // [rsp-10h] [rbp-90h]
  int v29; // [rsp+8h] [rbp-78h]
  bool v30; // [rsp+Ch] [rbp-74h]
  int v31; // [rsp+Ch] [rbp-74h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  bool v33; // [rsp+10h] [rbp-70h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  __int64 v35; // [rsp+18h] [rbp-68h]
  int v37; // [rsp+20h] [rbp-60h]
  int v38; // [rsp+28h] [rbp-58h]
  int v39; // [rsp+28h] [rbp-58h]
  char v42[52]; // [rsp+4Ch] [rbp-34h] BYREF

  v6 = *(char **)(a2 + 8);
  if ( !v6 )
  {
    v6 = "this";
    if ( (*(_BYTE *)(a2 + 172) & 1) == 0 )
      v6 = (char *)byte_3F871B3;
  }
  sub_93ED80(*(_DWORD *)(a2 + 64), v42);
  v7 = (const char *)sub_93EC00((__int64)v6, a2);
  v38 = sub_9405D0(a1, *(_DWORD *)(a2 + 64), v8, v9, v10, v11);
  v12 = sub_941B90(a1, *(_QWORD *)(a2 + 120));
  v13 = *(_QWORD *)(a1 + 512);
  if ( v13 == *(_QWORD *)(a1 + 520) )
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 536) - 8LL) + 512LL;
  v14 = *(_QWORD *)(v13 - 8);
  v15 = *(_DWORD *)v42;
  v16 = unk_4D04660 != 0;
  if ( BYTE4(a4) )
  {
    v17 = a4;
    v18 = 0;
    if ( v7 )
    {
      v29 = *(_DWORD *)v42;
      v30 = unk_4D04660 != 0;
      v32 = v12;
      v19 = strlen(v7);
      v15 = v29;
      v16 = v30;
      v12 = v32;
      v17 = a4;
      v18 = v19;
    }
    v20 = sub_ADF7F0((int)a1 + 16, v14, (_DWORD)v7, v18, v17, v38, v15, v12, v16, 0, 0);
  }
  else
  {
    v26 = 0;
    if ( v7 )
    {
      v31 = *(_DWORD *)v42;
      v33 = unk_4D04660 != 0;
      v35 = v12;
      v27 = strlen(v7);
      v15 = v31;
      v16 = v33;
      v12 = v35;
      v26 = v27;
    }
    v20 = sub_ADFB30((int)a1 + 16, v14, (_DWORD)v7, v26, v38, v15, v12, v16, 0, 0);
  }
  v21 = *(_QWORD *)(a5 + 48);
  v37 = *(unsigned __int16 *)(a2 + 68);
  v39 = *(_DWORD *)v42;
  v34 = v21;
  v22 = sub_BD5C60(a3, *(unsigned int *)v42, *(unsigned __int16 *)(a2 + 68));
  v23 = sub_B01860(v22, v39, v37, v14, 0, 0, 0, 1);
  v24 = sub_ADD5E0(a1 + 16, 0, 0);
  sub_ADF070(a1 + 16, a3, v20, v24, v23, v34);
  *(_DWORD *)(a1 + 456) = *(_DWORD *)(a2 + 64);
  *(_WORD *)(a1 + 460) = *(_WORD *)(a2 + 68);
  return v28;
}
