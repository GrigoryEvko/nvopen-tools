// Function: sub_80E180
// Address: 0x80e180
//
__int64 __fastcall sub_80E180(__int64 a1, __int64 *a2)
{
  char v3; // al
  __int64 v4; // r13
  __int64 v5; // rdi
  __int64 v6; // rax
  bool v7; // zf
  const __m128i *v8; // rdi
  char *v10; // rdi
  __int64 v11; // r15
  unsigned __int64 v12; // r13
  __int64 v13; // rax
  _DWORD v14[13]; // [rsp+Ch] [rbp-34h] BYREF

  while ( 1 )
  {
    v3 = *(_BYTE *)(a1 + 173);
    v4 = *(_QWORD *)(a1 + 120);
    if ( v3 == 11 )
      return v4;
    if ( v3 != 13 )
    {
      sub_80D8A0((const __m128i *)a1, 1u, 0, a2);
      return v4;
    }
    v5 = qword_4F18BE0;
    v6 = *a2 + 2;
    if ( (*(_BYTE *)(a1 + 176) & 1) != 0 )
    {
      *a2 = v6;
      sub_8238B0(v5, "di", 2);
      if ( (*(_BYTE *)(a1 + 176) & 2) != 0 )
        v10 = *(char **)(a1 + 184);
      else
        v10 = *(char **)(*(_QWORD *)(a1 + 184) + 8LL);
      sub_80BC40(v10, a2);
    }
    else
    {
      v7 = *(_BYTE *)(v4 + 173) == 11;
      *a2 = v6;
      if ( v7 )
      {
        sub_8238B0(v5, &unk_3C1B998, 2);
        v11 = *(_QWORD *)(a1 + 120);
        if ( (*(_BYTE *)(a1 + 176) & 2) != 0 )
        {
          sub_80D8A0(*(const __m128i **)(a1 + 184), 1u, 0, a2);
          if ( v11 )
            goto LABEL_17;
        }
        else
        {
          v12 = *(_QWORD *)(a1 + 184);
          sub_80BDC0(v12, a2);
          if ( v11 )
          {
            v14[0] = 0;
            v13 = sub_620FD0(v11, v14);
            sub_80BDC0(v12 + v13, a2);
LABEL_17:
            sub_80E180(*(_QWORD *)(v11 + 176), a2);
            return *(_QWORD *)(v11 + 120);
          }
        }
      }
      else
      {
        sub_8238B0(v5, "dx", 2);
        v8 = *(const __m128i **)(a1 + 184);
        if ( (*(_BYTE *)(a1 + 176) & 2) != 0 )
          sub_80D8A0(v8, 1u, 0, a2);
        else
          sub_80BDC0((unsigned __int64)v8, a2);
      }
    }
    a1 = *(_QWORD *)(a1 + 120);
  }
}
