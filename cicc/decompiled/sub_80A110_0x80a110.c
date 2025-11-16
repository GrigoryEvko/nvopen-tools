// Function: sub_80A110
// Address: 0x80a110
//
__int64 __fastcall sub_80A110(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rbx
  __int64 v3; // rax
  _QWORD *v5; // rax
  char *v6; // r14
  size_t v7; // rax
  __int64 v8; // r12

  if ( *a1 )
    v2 = sub_72B7A0(a1);
  else
    v2 = (_QWORD *)qword_4D03FF0;
  v3 = v2[51];
  qword_4F18BB0 = v3;
  if ( !v3 )
  {
    qword_4F18BB0 = sub_823970(136);
    sub_726CF0((__m128i *)qword_4F18BB0, 0);
    v3 = qword_4F18BB0;
    v2[51] = qword_4F18BB0;
  }
  if ( !*(_QWORD *)(v3 + 8) )
  {
    if ( dword_4F18BD8 )
    {
      *(_DWORD *)(a2 + 48) = 1;
    }
    else
    {
      if ( *a1 )
        v5 = sub_72B7A0(a1);
      else
        v5 = (_QWORD *)qword_4D03FF0;
      v6 = *(char **)v5[49];
      if ( !v6 )
        v6 = sub_723F40(0);
      if ( !*(_DWORD *)(a2 + 48) )
      {
        v7 = strlen(v6);
        v8 = sub_823970(v7 + 10);
        *(_QWORD *)v8 = 0x414E5245544E495FLL;
        *(_BYTE *)(v8 + 8) = 76;
        strcpy((char *)(v8 + 9), v6);
        *(_QWORD *)(qword_4F18BB0 + 8) = v8;
      }
    }
  }
  return v2[51];
}
