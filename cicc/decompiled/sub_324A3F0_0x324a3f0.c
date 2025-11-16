// Function: sub_324A3F0
// Address: 0x324a3f0
//
void __fastcall sub_324A3F0(__int64 *a1, __int16 a2, __int64 a3)
{
  __int64 v3; // r8
  unsigned __int8 *v6; // rcx
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 *v11; // r14
  unsigned __int64 v12; // rdx
  __int64 *v13; // rax
  unsigned int v14; // edx
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rsi
  __int64 *v21; // rdi
  __int64 v22; // rax
  unsigned __int64 *v23[2]; // [rsp-D8h] [rbp-D8h] BYREF
  _DWORD v24[6]; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 *v25; // [rsp-B0h] [rbp-B0h]
  __int64 v26; // [rsp-A0h] [rbp-A0h] BYREF
  char v27; // [rsp-64h] [rbp-64h]
  __int64 **v28; // [rsp-58h] [rbp-58h]

  if ( !a3 )
    return;
  v3 = (a3 >> 1) & 3;
  if ( v3 == 1 )
  {
    if ( (a3 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v6 = sub_3247C80(*a1, (unsigned __int8 *)(a3 & 0xFFFFFFFFFFFFFFF8LL));
      if ( v6 )
        sub_32494F0((__int64 *)*a1, a1[1], a2, (unsigned __int64)v6);
    }
    return;
  }
  if ( v3 == 2 )
  {
    v7 = a3 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (a3 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v8 = sub_A777F0(0x10u, (__int64 *)(*a1 + 88));
      v9 = v8;
      if ( v8 )
      {
        *(_QWORD *)v8 = 0;
        *(_DWORD *)(v8 + 8) = 0;
      }
      v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)*a1 + 72LL))(*a1);
      sub_3247620((__int64)v24, *(_QWORD *)(*a1 + 184), v10, v9);
      v27 = v27 & 0xF8 | 2;
      v23[0] = *(unsigned __int64 **)(v7 + 16);
      v23[1] = *(unsigned __int64 **)(v7 + 24);
      sub_3244870(v24, v23);
      v11 = (__int64 *)*a1;
      sub_3243D40((__int64)v24);
      sub_3249620(v11, a1[1], a2, v28);
      if ( v25 != &v26 )
        _libc_free((unsigned __int64)v25);
    }
    return;
  }
  if ( v3 )
    return;
  v12 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v12 )
    return;
  v13 = *(__int64 **)(v12 + 24);
  v14 = *(_DWORD *)(v12 + 32);
  if ( a2 != 8965 )
  {
    if ( a2 != 34 || (v15 = *(_QWORD *)a1[2], v15 == -1) )
    {
      v17 = (__int64 *)*a1;
      if ( v14 <= 0x40 )
      {
LABEL_23:
        if ( v14 )
          v3 = (__int64)((_QWORD)v13 << (64 - (unsigned __int8)v14)) >> (64 - (unsigned __int8)v14);
        goto LABEL_25;
      }
      v22 = *v13;
    }
    else
    {
      if ( v14 <= 0x40 )
      {
        v16 = 0;
        if ( v14 )
          v16 = (__int64)((_QWORD)v13 << (64 - (unsigned __int8)v14)) >> (64 - (unsigned __int8)v14);
        if ( v15 == v16 && *(_BYTE *)a1[3] )
          return;
        v17 = (__int64 *)*a1;
        goto LABEL_23;
      }
      v22 = *v13;
      v3 = v22;
      if ( v15 != v22 )
      {
        v17 = (__int64 *)*a1;
        goto LABEL_25;
      }
      if ( *(_BYTE *)a1[3] )
        return;
      v17 = (__int64 *)*a1;
    }
    v3 = v22;
LABEL_25:
    v18 = a1[1];
    v24[0] = 65549;
    sub_32498F0(v17, (unsigned __int64 **)(v18 + 8), a2, 65549, v3);
    return;
  }
  if ( v14 <= 0x40 )
  {
    if ( !v14 )
      return;
    v19 = (__int64)((_QWORD)v13 << (64 - (unsigned __int8)v14)) >> (64 - (unsigned __int8)v14);
  }
  else
  {
    v19 = *v13;
  }
  if ( v19 )
  {
    v20 = a1[1];
    v21 = (__int64 *)*a1;
    v24[0] = 65549;
    sub_3249A20(v21, (unsigned __int64 **)(v20 + 8), 8965, 65549, v19);
  }
}
