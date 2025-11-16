// Function: sub_324F7C0
// Address: 0x324f7c0
//
__int64 __fastcall sub_324F7C0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int8 v3; // al
  __int64 v4; // rcx
  __int64 v5; // r15
  __int64 v6; // r14
  __int16 v8; // ax
  unsigned __int8 v9; // al
  __int64 v10; // rbx
  char *v11; // r15
  const void *v12; // rcx
  size_t v13; // rdx
  size_t v14; // r8
  int v15; // edx
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  unsigned int v18; // eax

  if ( !a2 )
    return 0;
  v2 = a2 - 16;
  v3 = *(_BYTE *)(a2 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(_QWORD *)(a2 - 32);
  else
    v4 = v2 - 8LL * ((v3 >> 2) & 0xF);
  v5 = (*(__int64 (__fastcall **)(__int64 *, _QWORD))(*a1 + 48))(a1, *(_QWORD *)(v4 + 8));
  v6 = (__int64)sub_3247C80((__int64)a1, (unsigned __int8 *)a2);
  if ( !v6 )
  {
    v8 = sub_AF18C0(a2);
    v6 = sub_324C6D0(a1, v8, v5, (unsigned __int8 *)a2);
    v9 = *(_BYTE *)(a2 - 16);
    if ( (v9 & 2) != 0 )
      v10 = *(_QWORD *)(a2 - 32);
    else
      v10 = v2 - 8LL * ((v9 >> 2) & 0xF);
    v11 = *(char **)(v10 + 24);
    v12 = *(const void **)(v10 + 16);
    if ( v12 )
    {
      v12 = (const void *)sub_B91420(*(_QWORD *)(v10 + 16));
      v14 = v13;
    }
    else
    {
      v14 = 0;
    }
    sub_324AD70(a1, v6, 3, v12, v14);
    sub_32495E0(a1, v6, (__int64)v11, 73);
    sub_3249E10(a1, v6, a2);
    sub_3249FA0(a1, v6, 63);
    sub_3249FA0(a1, v6, 60);
    v15 = *(_DWORD *)(a2 + 20);
    if ( (v15 & 0x40) != 0 )
    {
      sub_3249FA0(a1, v6, 52);
      v15 = *(_DWORD *)(a2 + 20);
    }
    sub_3249F00(a1, v6, v15);
    v16 = (_BYTE *)sub_AF2DC0(a2);
    if ( v16 && *v16 == 17 )
      sub_324A3E0(a1, v6, (__int64)v16, v11);
    v17 = (_BYTE *)sub_AF2DC0(a2);
    if ( v17 && *v17 == 18 )
      sub_324A320(a1, v6, (__int64)v17);
    if ( (unsigned __int16)sub_3220AA0(a1[26]) > 4u )
    {
      v18 = (unsigned int)sub_AF18D0(a2) >> 3;
      if ( v18 )
        sub_3249A20(a1, (unsigned __int64 **)(v6 + 8), 136, 65551, v18 & 0x1FFFFFFF);
    }
  }
  return v6;
}
