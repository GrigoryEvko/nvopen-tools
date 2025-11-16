// Function: sub_72CAB0
// Address: 0x72cab0
//
__int64 __fastcall sub_72CAB0(void *a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  char v3; // al
  __int64 v4; // r13
  char v6; // al

  v1 = sub_8865A0(a1);
  if ( v1 )
  {
    v2 = v1;
    v3 = *(_BYTE *)(v1 + 80);
    if ( v3 != 6 )
    {
      if ( v3 != 3 )
        goto LABEL_4;
      if ( !(unsigned int)sub_8D2870(*(_QWORD *)(v2 + 88)) )
      {
        v6 = *(_BYTE *)(v2 + 80);
        v4 = *(_QWORD *)(v2 + 88);
        if ( v6 == 3 )
        {
LABEL_5:
          if ( (*(_BYTE *)(v4 + 140) & 0xFB) != 8 )
            return v4;
LABEL_12:
          if ( (unsigned int)sub_8D4C10(v4, dword_4F077C4 != 2) )
          {
            v4 = sub_72C930();
            sub_6E5E30(0xB64u, dword_4F07508, (__int64)a1);
          }
          return v4;
        }
        if ( v6 == 6 )
        {
          if ( (*(_BYTE *)(v4 + 140) & 0xFB) != 8 )
            return v4;
          goto LABEL_12;
        }
LABEL_4:
        v4 = *(_QWORD *)(v2 + 88);
        goto LABEL_5;
      }
    }
  }
  sub_6E5E30(0xB64u, dword_4F07508, (__int64)a1);
  return sub_72C930();
}
