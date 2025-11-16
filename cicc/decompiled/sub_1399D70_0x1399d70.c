// Function: sub_1399D70
// Address: 0x1399d70
//
__int64 __fastcall sub_1399D70(char *s, char *a2)
{
  __int64 v3; // r12
  size_t v4; // rax
  __int64 *v5; // rdi
  __int64 v6; // rax

  v3 = sub_22077B0(192);
  if ( v3 )
  {
    v4 = 0;
    if ( s )
      v4 = strlen(s);
    *(_QWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 80) = v3 + 64;
    v5 = (__int64 *)(v3 + 160);
    *(_QWORD *)(v3 + 88) = v3 + 64;
    *(_QWORD *)(v3 + 128) = v3 + 112;
    *(_QWORD *)(v3 + 136) = v3 + 112;
    *(_QWORD *)(v3 + 16) = &unk_4F98AA0;
    *(_DWORD *)(v3 + 24) = 5;
    *(_QWORD *)v3 = &unk_49E90F0;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_DWORD *)(v3 + 64) = 0;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 96) = 0;
    *(_DWORD *)(v3 + 112) = 0;
    *(_QWORD *)(v3 + 120) = 0;
    *(_QWORD *)(v3 + 144) = 0;
    *(_BYTE *)(v3 + 152) = 0;
    *(_QWORD *)(v3 + 160) = v3 + 176;
    if ( s )
    {
      a2 = s;
      sub_1399600(v5, s, (__int64)&s[v4]);
    }
    else
    {
      *(_QWORD *)(v3 + 168) = 0;
      *(_BYTE *)(v3 + 176) = 0;
    }
    *(_QWORD *)v3 = off_49E9198;
    v6 = sub_163A1D0(v5, a2);
    sub_1399970(v6);
  }
  return v3;
}
