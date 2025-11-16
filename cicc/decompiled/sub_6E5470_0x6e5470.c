// Function: sub_6E5470
// Address: 0x6e5470
//
__int64 __fastcall sub_6E5470(__int64 a1, _DWORD *a2)
{
  int v2; // r13d
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 *v6; // rbx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 result; // rax
  __int64 v10; // r12
  __int64 v11; // r15
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // rdi
  unsigned int v20; // [rsp+14h] [rbp-4Ch] BYREF
  __int64 v21; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v22[8]; // [rsp+20h] [rbp-40h] BYREF

  v2 = 0;
  v3 = a1;
  v6 = (__int64 *)sub_736C60(19, a1);
  do
  {
    v10 = v6[4];
    if ( !v10 || *(_BYTE *)(v10 + 10) != 5 )
    {
LABEL_2:
      v2 = 1;
      goto LABEL_3;
    }
    v11 = *(_QWORD *)(v10 + 40);
    if ( *(_BYTE *)(v11 + 24) == 2 )
    {
      v12 = *(_QWORD *)(v11 + 56);
    }
    else
    {
      v13 = sub_724DC0(19, v3, v4, v5, v7, v8);
      v3 = 1;
      v21 = v13;
      if ( !(unsigned int)sub_7A30C0(v11, 1, 1, v13) )
      {
        v2 = 1;
        sub_67E3D0(v22);
        sub_724E30(&v21);
        goto LABEL_3;
      }
      sub_7296C0(&v20);
      v12 = sub_724E50(&v21, 1, v14, v15, v16);
      v17 = sub_73A720(v12);
      v18 = v20;
      *(_QWORD *)(v10 + 40) = v17;
      sub_729730(v18);
      sub_67E3D0(v22);
      sub_724E30(&v21);
    }
    if ( !v12 || (unsigned int)sub_711520(v12, v3) )
      goto LABEL_2;
LABEL_3:
    v3 = *v6;
    result = sub_736C60(19, *v6);
    v6 = (__int64 *)result;
  }
  while ( result );
  if ( v2 )
  {
    result = sub_6E5430();
    if ( (_DWORD)result )
      return sub_6851C0(0xB00u, a2);
  }
  return result;
}
