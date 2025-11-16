// Function: sub_813140
// Address: 0x813140
//
void __fastcall sub_813140(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r15d
  __int64 v7; // r13
  __int64 v8; // r12
  char v9; // bl
  __int64 v10; // rax
  _QWORD *v11; // r14
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // [rsp+0h] [rbp-80h] BYREF
  __int64 v20; // [rsp+8h] [rbp-78h]
  __int64 v21; // [rsp+18h] [rbp-68h]
  char v22; // [rsp+20h] [rbp-60h]
  __int64 v23; // [rsp+28h] [rbp-58h]
  __int64 v24; // [rsp+30h] [rbp-50h]
  int v25; // [rsp+38h] [rbp-48h]
  char v26; // [rsp+3Ch] [rbp-44h]
  __int64 v27; // [rsp+40h] [rbp-40h]

  v6 = (int)a3;
  v7 = a4;
  v8 = a1;
  v9 = a2;
  if ( (_BYTE)a2 == 7 )
  {
    if ( (*(_BYTE *)(a1 + 172) & 2) != 0 )
    {
      a1 = *(_QWORD *)(a1 + 120);
      a2 = (__int64)&v19;
      v18 = sub_808FB0(a1, &v19);
      if ( v18 )
        *(_QWORD *)(v8 + 8) = v18;
    }
    if ( unk_4D04170 )
    {
      a2 = 11;
      a1 = v7;
      v20 = 0;
      sub_737670(v7, 0xBu, (__int64 (__fastcall *)(__int64, _QWORD, _DWORD *))sub_80A3E0, &v19, 12);
      v11 = (_QWORD *)v20;
      if ( v20 )
      {
        do
        {
          a1 = v11[1];
          sub_80AD10(a1);
          v11 = (_QWORD *)*v11;
        }
        while ( v11 );
        a4 = v20;
        if ( v20 )
        {
          v12 = (__int64 *)qword_4F18B98;
          if ( qword_4F18B98 )
          {
            do
            {
              a3 = v12;
              v12 = (__int64 *)*v12;
            }
            while ( v12 );
            *a3 = v20;
          }
          else
          {
            qword_4F18B98 = v20;
          }
        }
      }
      v13 = *(_QWORD *)(v8 + 104);
      if ( !v13 || (*(_BYTE *)(v13 + 11) & 0x20) == 0 )
      {
        a2 = 7;
        a1 = v8;
        sub_80A450(v8, 7u);
      }
    }
  }
  if ( (*(_BYTE *)(v8 + 89) & 8) == 0
    && (*(_QWORD *)(v8 + 8) || v9 == 7 && (*(_BYTE *)(v8 + 170) & 2) != 0 && *(_BYTE *)(v8 + 136) != 3) )
  {
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    sub_809110(a1, a2, a3, a4, a5, a6, 0, 0, 0);
    sub_823800(qword_4F18BE0);
    v19 += 2;
    sub_8238B0(qword_4F18BE0, &unk_3C1BC40, 2);
    if ( (*(_BYTE *)(v7 + 91) & 1) != 0 || sub_736A10(v7) )
      sub_80BD00((_QWORD *)v7, (__int64)&v19);
    sub_811640(v7, &v19);
    v10 = *(_QWORD *)(v8 + 40);
    if ( v10 && *(_BYTE *)(v10 + 28) == 16 )
    {
      v14 = qword_4F18BE0;
      ++v19;
      v15 = *(_QWORD *)(qword_4F18BE0 + 16);
      if ( (unsigned __int64)(v15 + 1) > *(_QWORD *)(qword_4F18BE0 + 8) )
      {
        sub_823810(qword_4F18BE0);
        v14 = qword_4F18BE0;
        v15 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(*(_QWORD *)(v14 + 32) + v15) = 78;
      ++*(_QWORD *)(v14 + 16);
      sub_810560(*(_QWORD *)(*(_QWORD *)(v8 + 40) + 32LL), &v19);
      sub_80BC40(*(char **)(v8 + 8), &v19);
      v16 = qword_4F18BE0;
      ++v19;
      v17 = *(_QWORD *)(qword_4F18BE0 + 16);
      if ( (unsigned __int64)(v17 + 1) > *(_QWORD *)(qword_4F18BE0 + 8) )
      {
        sub_823810(qword_4F18BE0);
        v16 = qword_4F18BE0;
        v17 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(*(_QWORD *)(v16 + 32) + v17) = 69;
      ++*(_QWORD *)(v16 + 16);
    }
    else if ( v9 == 7 )
    {
      if ( (*(_BYTE *)(v8 + 170) & 2) != 0 && *(_BYTE *)(v8 + 136) != 3 )
      {
        sub_812EE0(v8, &v19);
      }
      else
      {
        sub_80BC40(*(char **)(v8 + 8), &v19);
        if ( *(char *)(v8 + 168) < 0 )
          sub_80B920(*(__int64 **)(v8 + 104), &v19);
      }
    }
    else
    {
      sub_80BC40(*(char **)(v8 + 8), &v19);
    }
    sub_80C040((__int64 *)v8, &v19);
    sub_80B290(v8, v6, (__int64)&v19);
  }
}
