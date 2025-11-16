// Function: sub_F8E330
// Address: 0xf8e330
//
__int64 __fastcall sub_F8E330(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdi
  _QWORD v7[2]; // [rsp+0h] [rbp-10h] BYREF
  __int64 savedregs; // [rsp+10h] [rbp+0h] BYREF

  v2 = 0;
  v7[0] = a1;
  v3 = v7;
  v7[1] = a2;
  while ( 1 )
  {
    if ( a1 )
    {
      v4 = *(_QWORD *)(a1 + 56);
      v5 = a1 + 48;
      if ( v5 != v4 )
        break;
    }
LABEL_9:
    if ( ++v3 == &savedregs )
      return v2;
    a1 = *v3;
  }
  while ( 1 )
  {
    if ( !v4 )
      BUG();
    if ( *(_BYTE *)(v4 - 24) != 62 )
      goto LABEL_8;
    if ( v2 )
      return 0;
    v2 = v4 - 24;
LABEL_8:
    v4 = *(_QWORD *)(v4 + 8);
    if ( v5 == v4 )
      goto LABEL_9;
  }
}
