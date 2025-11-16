// Function: sub_7FBA90
// Address: 0x7fba90
//
__int64 __fastcall sub_7FBA90(__int64 a1, __int64 a2, unsigned int a3, __m128i *a4)
{
  __int64 i; // r12
  __int64 result; // rax
  int v8; // edx
  _QWORD *v9; // r15
  __int64 *v10; // rax
  __int64 v11; // [rsp-10h] [rbp-50h]
  char v12; // [rsp+Ch] [rbp-34h]

  for ( i = sub_7F9140(a2); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( *(_BYTE *)(a2 + 16) || (result = sub_8D4070(i), (_DWORD)result) )
  {
    v8 = sub_8D3410(i);
    if ( v8 )
    {
      if ( (unsigned int)sub_8D23E0(i) || (unsigned int)sub_8D4070(i) )
      {
        sub_7E3130(i);
        goto LABEL_7;
      }
      LOBYTE(v8) = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 128LL) + 176LL) < *(_QWORD *)(i + 176);
    }
    v12 = v8;
    result = sub_7E3130(i);
    if ( !(_DWORD)result && *(_BYTE *)(a1 + 48) != 6 && (v12 & 1) == 0 )
      return result;
LABEL_7:
    v9 = *(_QWORD **)(a2 + 56);
    v10 = (__int64 *)sub_7F98A0(a2, 1);
    sub_7FB7C0(i, a3, v10, v9, 0, 0, a4);
    return v11;
  }
  return result;
}
