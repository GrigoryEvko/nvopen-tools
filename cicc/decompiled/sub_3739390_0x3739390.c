// Function: sub_3739390
// Address: 0x3739390
//
void __fastcall sub_3739390(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  unsigned __int64 v5; // rcx
  __int64 v6; // rcx
  __int64 v7; // rax
  unsigned __int8 v8; // dl
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  int v13; // eax

  v3 = sub_37362B0((__int64)a1, *(_QWORD *)(a2 + 8));
  v4 = *(_QWORD *)(a2 + 24);
  if ( v3 && (v5 = *(_QWORD *)(v3 + 24)) != 0 )
  {
    sub_32494F0(a1, *(_QWORD *)(a2 + 24), 49, v5);
    if ( *(_DWORD *)(a2 + 32) != 1 )
      return;
  }
  else
  {
    v13 = *(_DWORD *)(a2 + 32);
    if ( !v13 )
    {
      sub_3736380(a1, a2, *(_QWORD *)(a2 + 24));
      return;
    }
    if ( v13 != 1 )
      BUG();
    sub_3736500(a1, a2, *(_QWORD *)(a2 + 24));
  }
  v6 = *(_QWORD *)(a2 + 40);
  if ( v6 )
  {
    sub_3738C10(a1, v4, 17, v6);
    v7 = *(_QWORD *)(a2 + 8);
    v8 = *(_BYTE *)(v7 - 16);
    if ( (v8 & 2) != 0 )
      v9 = *(_QWORD *)(v7 - 32);
    else
      v9 = v7 - 16 - 8LL * ((v8 >> 2) & 0xF);
    v10 = *(_QWORD *)(v9 + 8);
    if ( v10 )
    {
      v11 = sub_B91420(v10);
      if ( v12 )
        sub_3237930(a1[26], (__int64)a1, *(_DWORD *)(a1[10] + 36), v11, v12, v4);
    }
  }
}
