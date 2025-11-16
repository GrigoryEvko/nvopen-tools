// Function: sub_6907F0
// Address: 0x6907f0
//
__int64 __fastcall sub_6907F0(__int64 *a1, __int64 *a2, unsigned __int16 a3, _DWORD *a4, int a5, __int64 a6)
{
  __int64 v8; // rax
  bool v9; // bl
  int v10; // edx
  __int64 v11; // rax
  __int64 result; // rax
  unsigned int v13; // eax
  _DWORD v16[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v8 = unk_4D03C50;
  v16[0] = 0;
  v9 = (*(_BYTE *)(unk_4D03C50 + 20LL) & 0x10) != 0;
  if ( dword_4F077C0 && *(_BYTE *)(unk_4D03C50 + 16LL) == 3 )
  {
    *(_BYTE *)(unk_4D03C50 + 20LL) |= 0x10u;
    if ( dword_4F077C4 != 2 )
    {
LABEL_5:
      sub_6F69D0(a1, 0);
      if ( !(unsigned int)sub_8D2D50(*a1)
        && (!HIDWORD(qword_4F077B4) || !(unsigned int)sub_8D2B80(*a1))
        && !(unsigned int)sub_8D2660(*a1)
        && !(unsigned int)sub_8D3D10(*a1)
        && !(unsigned int)sub_8D2630(*a1, 0) )
      {
        v13 = sub_6E9530();
        sub_6FB4D0(a1, v13);
      }
      sub_6F69D0(a2, 0);
      sub_68F410(a1, a2, a3, a5, a4, a6);
      goto LABEL_7;
    }
  }
  else if ( dword_4F077C4 != 2 )
  {
LABEL_4:
    if ( *(_BYTE *)(v8 + 16) == 2 )
    {
      sub_68BB70(a1, a2, a4, a6, v16);
      if ( v16[0] )
        goto LABEL_7;
    }
    goto LABEL_5;
  }
  if ( (unsigned int)sub_68FE10(a1, 0, 1) || (unsigned int)sub_68FE10(a2, 0, 1) )
    sub_84EC30(byte_4B6D300[a3], 0, 0, 1, 0, (_DWORD)a1, (__int64)a2, (__int64)a4, a5, 0, 0, a6, 0, 0, (__int64)v16);
  if ( !v16[0] )
  {
    v8 = unk_4D03C50;
    goto LABEL_4;
  }
LABEL_7:
  v10 = *((_DWORD *)a1 + 17);
  *(_WORD *)(a6 + 72) = *((_WORD *)a1 + 36);
  *(_DWORD *)(a6 + 68) = v10;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a6 + 68);
  v11 = *(__int64 *)((char *)a2 + 76);
  *(_QWORD *)(a6 + 76) = v11;
  unk_4F061D8 = v11;
  sub_6E3280(a6, a4);
  result = *(_BYTE *)(unk_4D03C50 + 20LL) & 0xEF;
  *(_BYTE *)(unk_4D03C50 + 20LL) = *(_BYTE *)(unk_4D03C50 + 20LL) & 0xEF | (16 * v9);
  return result;
}
