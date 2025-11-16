// Function: sub_154AC80
// Address: 0x154ac80
//
void __fastcall sub_154AC80(__int64 *a1, const char *a2, size_t a3, __int64 a4, __int64 a5, char a6)
{
  const char *v6; // r10
  __int64 v11; // r12
  _BYTE *v12; // rdi
  unsigned __int64 v13; // rax
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax

  v6 = a2;
  if ( !a6 || a5 )
  {
    v11 = *a1;
    if ( *((_BYTE *)a1 + 8) )
    {
      *((_BYTE *)a1 + 8) = 0;
    }
    else
    {
      v16 = sub_1263B40(*a1, (const char *)a1[2]);
      v6 = a2;
      v11 = v16;
    }
    v12 = *(_BYTE **)(v11 + 24);
    v13 = *(_QWORD *)(v11 + 16) - (_QWORD)v12;
    if ( v13 < a3 )
    {
      v17 = sub_16E7EE0(v11, v6, a3);
      v12 = *(_BYTE **)(v17 + 24);
      v11 = v17;
      v13 = *(_QWORD *)(v17 + 16) - (_QWORD)v12;
    }
    else if ( a3 )
    {
      memcpy(v12, v6, a3);
      v18 = *(_QWORD *)(v11 + 16);
      v12 = (_BYTE *)(a3 + *(_QWORD *)(v11 + 24));
      *(_QWORD *)(v11 + 24) = v12;
      v13 = v18 - (_QWORD)v12;
    }
    if ( v13 <= 2 )
    {
      sub_16E7EE0(v11, ": \"", 3);
    }
    else
    {
      v12[2] = 34;
      *(_WORD *)v12 = 8250;
      *(_QWORD *)(v11 + 24) += 3LL;
    }
    sub_16D16F0(a4, a5, *a1);
    v14 = *a1;
    v15 = *(_BYTE **)(*a1 + 24);
    if ( *(_BYTE **)(*a1 + 16) == v15 )
    {
      sub_16E7EE0(v14, "\"", 1);
    }
    else
    {
      *v15 = 34;
      ++*(_QWORD *)(v14 + 24);
    }
  }
}
