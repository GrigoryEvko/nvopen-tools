// Function: sub_ABD750
// Address: 0xabd750
//
__int64 __fastcall sub_ABD750(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 *v4; // rax
  unsigned int v5; // ebx
  bool v6; // dl
  __int64 *v7; // rax
  unsigned int v8; // ebx
  bool v9; // dl
  __int64 *v10; // rax
  unsigned int v11; // ebx
  bool v12; // dl

  if ( a2 > 0x173 )
    goto LABEL_31;
  if ( a2 > 0x136 )
  {
    switch ( a2 )
    {
      case 0x137u:
        sub_AB9F50(a1, a3, a3 + 32);
        return a1;
      case 0x149u:
        sub_AB5F70(a1, a3, a3 + 32);
        return a1;
      case 0x14Au:
        sub_AB64F0(a1, a3, a3 + 32);
        return a1;
      case 0x152u:
        sub_ABA520(a1, a3, a3 + 32);
        return a1;
      case 0x167u:
        sub_AB9DC0(a1, a3, a3 + 32);
        return a1;
      case 0x16Du:
        sub_AB6230(a1, a3, a3 + 32);
        return a1;
      case 0x16Eu:
        sub_AB6790(a1, a3, a3 + 32);
        return a1;
      case 0x173u:
        sub_ABA390(a1, a3, a3 + 32);
        return a1;
      default:
        goto LABEL_31;
    }
  }
  if ( a2 == 66 )
  {
    sub_ABD4A0(a1, a3);
    return a1;
  }
  if ( a2 > 0x42 )
  {
    if ( a2 == 67 )
    {
      v4 = sub_9876C0((__int64 *)(a3 + 32));
      v5 = *((_DWORD *)v4 + 2);
      if ( v5 <= 0x40 )
        v6 = *v4 == 0;
      else
        v6 = v5 == (unsigned int)sub_C444A0(v4);
      sub_ABD110(a1, a3, !v6);
      return a1;
    }
LABEL_31:
    BUG();
  }
  if ( a2 == 1 )
  {
    v7 = sub_9876C0((__int64 *)(a3 + 32));
    v8 = *((_DWORD *)v7 + 2);
    if ( v8 <= 0x40 )
      v9 = *v7 == 0;
    else
      v9 = v8 == (unsigned int)sub_C444A0(v7);
    sub_ABBBB0(a1, a3, !v9);
  }
  else
  {
    if ( a2 != 65 )
      goto LABEL_31;
    v10 = sub_9876C0((__int64 *)(a3 + 32));
    v11 = *((_DWORD *)v10 + 2);
    if ( v11 <= 0x40 )
      v12 = *v10 == 0;
    else
      v12 = v11 == (unsigned int)sub_C444A0(v10);
    sub_ABCC80(a1, a3, !v12);
  }
  return a1;
}
