// Function: sub_72F3C0
// Address: 0x72f3c0
//
__int64 __fastcall sub_72F3C0(__int64 a1, __int64 a2, char *a3, int a4, int a5)
{
  char *v6; // rbx
  __int64 ***v7; // r14
  __int64 **v8; // rdi
  int v9; // eax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r15
  __int64 i; // rdi
  __int64 **v15; // rax
  int v16; // eax
  char v17; // [rsp+Ch] [rbp-34h] BYREF

  v6 = a3;
  if ( !a3 )
    v6 = &v17;
  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  v7 = **(__int64 *****)(a1 + 168);
  if ( !v7 )
    return 0;
  v8 = v7[1];
  v9 = a4 ? sub_8D32E0(v8) : sub_8D30C0(v8);
  if ( !v9 || *v7 && ((_BYTE)(*v7)[4] & 4) == 0 )
    return 0;
  v13 = sub_8D46C0(v7[1]);
  for ( i = v13; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( a2 != i && !(unsigned int)sub_8D97D0(i, a2, 0, v11, v12) )
    return 0;
  if ( !a5 )
  {
    v15 = *v7;
    if ( *v7 )
    {
      while ( v15[5] || ((_BYTE)v15[4] & 0x10) != 0 )
      {
        v15 = (__int64 **)*v15;
        if ( !v15 )
          goto LABEL_24;
      }
      return 0;
    }
  }
LABEL_24:
  v16 = 0;
  if ( *(_BYTE *)(v13 + 140) == 12 )
    v16 = sub_8D4C10(v13, 1);
  *(_DWORD *)v6 = v16;
  return 1;
}
