// Function: sub_80FC70
// Address: 0x80fc70
//
__int64 __fastcall sub_80FC70(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rbx
  __int64 result; // rax
  __int64 v5; // r13
  _QWORD *v6; // rdi
  _QWORD *v7; // rdi
  _DWORD v8[13]; // [rsp+Ch] [rbp-34h] BYREF

  v3 = *(_QWORD **)a1;
  if ( *(_QWORD *)a1 )
  {
    do
    {
      while ( 1 )
      {
        v5 = v3[1];
        if ( v3[10] )
          break;
        result = sub_80F5E0(v3[1], 0, a2);
        v3 = (_QWORD *)*v3;
        if ( !v3 )
          goto LABEL_7;
      }
      result = sub_80C5A0(v3[1], 6, 1, 0, v8, a2);
      if ( !(_DWORD)result )
      {
        *a2 += 2LL;
        sub_8238B0(qword_4F18BE0, &unk_3C1BB44, 2);
        result = sub_80F5E0(v5, 0, a2);
        if ( !a2[5] )
          result = sub_80A250(v5, 6, 1, (__int64)a2);
      }
      v3 = (_QWORD *)*v3;
    }
    while ( v3 );
LABEL_7:
    if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
      goto LABEL_12;
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
      goto LABEL_12;
    v7 = (_QWORD *)qword_4F18BE0;
    ++*a2;
    result = v7[2];
    if ( (unsigned __int64)(result + 1) > v7[1] )
    {
      sub_823810(v7);
      v7 = (_QWORD *)qword_4F18BE0;
      result = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v7[4] + result) = 118;
    ++v7[2];
    if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
    {
LABEL_12:
      v6 = (_QWORD *)qword_4F18BE0;
      ++*a2;
      result = v6[2];
      if ( (unsigned __int64)(result + 1) > v6[1] )
      {
        sub_823810(v6);
        v6 = (_QWORD *)qword_4F18BE0;
        result = *(_QWORD *)(qword_4F18BE0 + 16);
        *(_BYTE *)(*(_QWORD *)(qword_4F18BE0 + 32) + result) = 122;
      }
      else
      {
        *(_BYTE *)(v6[4] + result) = 122;
      }
      ++v6[2];
    }
  }
  return result;
}
