// Function: sub_8B10C0
// Address: 0x8b10c0
//
__int64 __fastcall sub_8B10C0(__int64 a1, unsigned int *a2)
{
  unsigned int v3; // r14d
  unsigned __int16 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  int v17; // esi
  unsigned int v18; // edx
  _QWORD *v19; // rdi
  __int64 result; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r12
  _DWORD v26[13]; // [rsp+Ch] [rbp-34h] BYREF

  v26[0] = 0;
  v3 = dword_4F063F8;
  v4 = word_4F063FC[0];
  if ( (unsigned __int16)sub_7BE840(0, 0) == 44 )
  {
    sub_7B8B50(0, 0, v5, v6, v7, v8);
    sub_6851C0(0xB5Du, &dword_4F063F8);
    return sub_7B8B50(0xB5Du, &dword_4F063F8, v21, v22, v23, v24);
  }
  else
  {
    sub_89EFB0(a1, (__int64)a2);
    sub_7BDB60(1);
    sub_7B8B50(1u, a2, v9, v10, v11, v12);
    sub_8AF6E0(a1);
    *(_DWORD *)(*(_QWORD *)(a1 + 192) + 8LL) = *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
    v13 = sub_727300();
    v15 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184);
    *((_DWORD *)v13 + 8) = v3;
    *((_WORD *)v13 + 18) = v4;
    v13[3] = v15;
    *v13 = *(_QWORD *)(a1 + 488);
    v16 = *(_QWORD *)(a1 + 192);
    *(_QWORD *)(a1 + 488) = v13;
    if ( v16 )
      *(_QWORD *)(v16 + 32) = v13;
    if ( word_4F06418[0] == 294 )
    {
      v25 = *(_QWORD *)(a1 + 488);
      *(_QWORD *)(v25 + 16) = sub_6DE780(0, (__int64)a2, v16, v14);
    }
    v17 = *(_DWORD *)(a1 + 156);
    v18 = dword_4F06650[0];
    v19 = qword_4F061C0;
    *(_DWORD *)(a1 + 160) = dword_4F06650[0];
    sub_7AE700((__int64)(v19 + 3), v17, v18, 0, a1 + 248);
    sub_8992E0(a1, *(__int64 **)(a1 + 192), 0, v26);
    result = dword_4F06650[0];
    *(_DWORD *)(a1 + 280) = dword_4F06650[0];
  }
  return result;
}
