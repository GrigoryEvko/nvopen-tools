// Function: sub_88A060
// Address: 0x88a060
//
__int64 sub_88A060()
{
  _QWORD *v0; // r12
  __int64 v2; // rbx
  unsigned int v3; // eax
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  __int64 v6[12]; // [rsp+0h] [rbp-60h] BYREF

  v0 = (_QWORD *)unk_4D04978;
  if ( !unk_4D04978 )
  {
    v2 = sub_878540("__va_list_tag", 0xDu, v6);
    v3 = dword_4D045F4;
    if ( dword_4D045F4 && (!dword_4D0455C || unk_4D04600 > 0x30DA3u) || unk_4D045F0 )
    {
      if ( v2 )
      {
        v0 = *(_QWORD **)(v2 + 88);
      }
      else
      {
        v0 = sub_603190();
        v3 = dword_4D045F4;
      }
      if ( v3 && (!dword_4D0455C || unk_4D04600 > 0x30DA3u) )
        return sub_72D2E0(v0);
    }
    else if ( unk_4F06A80 | HIDWORD(qword_4F06A78) | (unsigned int)qword_4F06A78 )
    {
      v0 = sub_7259C0(8);
      if ( v2 )
        v4 = *(_QWORD **)(v2 + 88);
      else
        v4 = sub_603190();
      v0[20] = v4;
      v0[22] = 1;
      sub_8D6090(v0);
    }
    else
    {
      if ( HIDWORD(qword_4F077B4) )
        v5 = sub_72BA30(0);
      else
        v5 = (_QWORD *)sub_72CBE0();
      return sub_72D2E0(v5);
    }
  }
  return (__int64)v0;
}
