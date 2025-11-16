// Function: sub_8128F0
// Address: 0x8128f0
//
void __fastcall sub_8128F0(__int64 a1, _QWORD *a2)
{
  char v2; // al
  _QWORD *v3; // rdi
  __int64 v4; // rax
  int v5; // [rsp+Ch] [rbp-34h] BYREF
  unsigned int v6[2]; // [rsp+10h] [rbp-30h] BYREF
  __int64 v7; // [rsp+18h] [rbp-28h]

  v5 = 0;
  if ( a1 )
  {
    v2 = *(_BYTE *)(a1 + 33);
    if ( *(_QWORD *)(a1 + 8) || (v2 & 1) != 0 )
    {
      v7 = *(_QWORD *)(a1 + 8);
      v6[1] = v2 & 1;
      v6[0] = 1;
      sub_812470(v6, &v5, 1u, a2);
      if ( v5 )
      {
        v3 = (_QWORD *)qword_4F18BE0;
        ++*a2;
        v4 = v3[2];
        if ( (unsigned __int64)(v4 + 1) > v3[1] )
        {
          sub_823810(v3);
          v3 = (_QWORD *)qword_4F18BE0;
          v4 = *(_QWORD *)(qword_4F18BE0 + 16);
        }
        *(_BYTE *)(v3[4] + v4) = 69;
        ++v3[2];
      }
    }
  }
}
