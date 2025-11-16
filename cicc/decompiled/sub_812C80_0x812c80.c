// Function: sub_812C80
// Address: 0x812c80
//
void __fastcall sub_812C80(__int64 a1, unsigned __int8 a2, __int64 a3, _QWORD *a4)
{
  char *v7; // rdi
  __int64 v8; // rax
  _QWORD *i; // r13
  char *v10; // rdi
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rax
  int v14; // [rsp+Ch] [rbp-34h] BYREF
  __int64 *v15; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v16[5]; // [rsp+18h] [rbp-28h] BYREF

  v14 = 0;
  sub_811730(a1, a2, &v14, (__int64 *)&v15, 0, (__int64)a4);
  if ( a2 == 7 )
  {
    if ( (*(_BYTE *)(a1 + 170) & 2) == 0 || *(_BYTE *)(a1 + 136) == 3 )
    {
      if ( (*(_BYTE *)(a1 + 176) & 0x20) == 0 )
        goto LABEL_2;
      *a4 += 3LL;
      sub_8238B0(qword_4F18BE0, "TAX", 3);
      sub_80D8A0(*(const __m128i **)(a1 + 184), 0, 0, a4);
    }
    else
    {
      *a4 += 2LL;
      sub_8238B0(qword_4F18BE0, "DC", 2);
      for ( i = *(_QWORD **)(a1 + 128); i; i = (_QWORD *)*i )
      {
        v11 = i[2];
        v10 = 0;
        if ( (*(_BYTE *)(v11 + 89) & 0x40) == 0 )
        {
          if ( (*(_BYTE *)(v11 + 89) & 8) != 0 )
            v10 = *(char **)(v11 + 24);
          else
            v10 = *(char **)(v11 + 8);
        }
        sub_80BC40(v10, a4);
      }
    }
    v12 = (_QWORD *)qword_4F18BE0;
    ++*a4;
    v13 = v12[2];
    if ( (unsigned __int64)(v13 + 1) > v12[1] )
    {
      sub_823810(v12);
      v12 = (_QWORD *)qword_4F18BE0;
      v13 = *(_QWORD *)(qword_4F18BE0 + 16);
    }
    *(_BYTE *)(v12[4] + v13) = 69;
    ++v12[2];
    goto LABEL_29;
  }
LABEL_2:
  if ( !a3 || !(unsigned int)sub_80C5A0(a3, 59, 0, 0, v16, a4) )
  {
    if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
      v7 = *(char **)(a1 + 24);
    else
      v7 = *(char **)(a1 + 8);
    sub_80BC40(v7, a4);
    if ( a3 && !a4[5] )
      sub_80A250(a3, 59, 0, (__int64)a4);
    v8 = *(_QWORD *)a1;
    if ( !*(_QWORD *)a1 || *(_BYTE *)(v8 + 80) != 2 || !sub_72AE00(*(_QWORD *)(v8 + 88)) )
      sub_80C040((__int64 *)a1, a4);
  }
  if ( a2 == 7 )
  {
LABEL_29:
    if ( *(char *)(a1 + 168) < 0 )
      sub_80B920(*(__int64 **)(a1 + 104), a4);
    if ( (*(_BYTE *)(a1 + 170) & 0x10) != 0 && **(_QWORD **)(a1 + 216) )
    {
      v16[0] = **(_QWORD **)(a1 + 216);
      sub_811CB0(v16, 0, 0, a4);
    }
  }
  sub_80C110(v14, v15, a4);
}
