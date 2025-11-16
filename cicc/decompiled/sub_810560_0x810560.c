// Function: sub_810560
// Address: 0x810560
//
void __fastcall sub_810560(__int64 a1, _QWORD *a2)
{
  const char *v2; // r13
  size_t v3; // rax
  __int64 v4; // rdx
  size_t v5; // rax
  _BYTE v6[96]; // [rsp+0h] [rbp-60h] BYREF

  if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
  {
    v2 = *(const char **)(a1 + 24);
    if ( v2 )
      goto LABEL_3;
LABEL_8:
    sub_80FE00(a1, (__int64)a2);
    if ( (*(_BYTE *)(a1 + 143) & 0x40) == 0 )
      return;
LABEL_9:
    sub_80B920(*(__int64 **)(a1 + 104), a2);
    return;
  }
  v2 = *(const char **)(a1 + 8);
  if ( !v2 )
    goto LABEL_8;
LABEL_3:
  v3 = strlen(v2);
  if ( v3 > 9 )
  {
    v4 = (int)sub_622470(v3, v6);
  }
  else
  {
    v6[1] = 0;
    v4 = 1;
    v6[0] = v3 + 48;
  }
  *a2 += v4;
  sub_8238B0(qword_4F18BE0, v6, v4);
  v5 = strlen(v2);
  *a2 += v5;
  sub_8238B0(qword_4F18BE0, v2, v5);
  if ( (*(_BYTE *)(a1 + 143) & 0x40) != 0 )
    goto LABEL_9;
}
