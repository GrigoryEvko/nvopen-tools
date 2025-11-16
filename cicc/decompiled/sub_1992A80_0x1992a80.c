// Function: sub_1992A80
// Address: 0x1992a80
//
bool __fastcall sub_1992A80(_DWORD *a1, _DWORD *a2, char a3)
{
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax
  _DWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  _DWORD *v12; // r8
  _DWORD *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  bool v16; // cf
  bool result; // al
  unsigned int v18; // edx
  unsigned int v19; // eax
  unsigned int v20; // ecx
  unsigned int v21; // esi
  unsigned int v22; // edx
  unsigned int v23; // eax
  unsigned int v24; // eax
  bool v25; // zf
  unsigned int v26; // ebx
  unsigned int v27; // ebx
  unsigned int v28; // ebx
  unsigned int v29; // eax
  unsigned int v30; // ecx
  unsigned int v31; // esi

  v6 = sub_16D5D50();
  v7 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v8 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v9 = v7[2];
        v10 = v7[3];
        if ( v6 <= v7[4] )
          break;
        v7 = (_QWORD *)v7[3];
        if ( !v10 )
          goto LABEL_6;
      }
      v8 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v9 );
LABEL_6:
    if ( v8 != dword_4FA0208 && v6 >= *((_QWORD *)v8 + 4) )
    {
      v11 = *((_QWORD *)v8 + 7);
      v12 = v8 + 12;
      if ( v11 )
      {
        v13 = v8 + 12;
        do
        {
          while ( 1 )
          {
            v14 = *(_QWORD *)(v11 + 16);
            v15 = *(_QWORD *)(v11 + 24);
            if ( *(_DWORD *)(v11 + 32) >= dword_4FB1D28 )
              break;
            v11 = *(_QWORD *)(v11 + 24);
            if ( !v15 )
              goto LABEL_13;
          }
          v13 = (_DWORD *)v11;
          v11 = *(_QWORD *)(v11 + 16);
        }
        while ( v14 );
LABEL_13:
        if ( v12 != v13 && dword_4FB1D28 >= v13[8] && (int)v13[9] > 0 )
        {
          if ( byte_4FB1DC0 )
          {
            v16 = *a1 < *a2;
            if ( *a1 != *a2 )
              return v16;
          }
        }
      }
    }
  }
  v18 = a1[1];
  v19 = a2[1];
  if ( !a3 )
  {
    v20 = a1[3];
    v21 = a2[3];
    v22 = a1[4] + v20 + a1[6] + v18;
    v23 = a2[4] + v21 + a2[6] + v19;
    v16 = v22 < v23;
    if ( v22 != v23 )
      return v16;
    v24 = a2[2];
    v16 = a1[2] < v24;
    if ( a1[2] != v24 )
      return v16;
    result = 1;
    v25 = v21 == v20;
    if ( v21 > v20 )
      return result;
    goto LABEL_26;
  }
  if ( v19 != v18 )
    return v19 > v18;
  v29 = a2[2];
  v16 = a1[2] < v29;
  if ( a1[2] != v29 )
    return v16;
  v30 = a1[3];
  v31 = a2[3];
  result = 1;
  v25 = v31 == v30;
  if ( v31 <= v30 )
  {
LABEL_26:
    result = 0;
    if ( v25 )
    {
      v26 = a2[4];
      result = 1;
      if ( a1[4] >= v26 )
      {
        result = 0;
        if ( a1[4] == v26 )
        {
          v27 = a2[7];
          result = 1;
          if ( a1[7] >= v27 )
          {
            result = 0;
            if ( a1[7] == v27 )
            {
              v28 = a2[5];
              result = 1;
              if ( a1[5] >= v28 )
              {
                result = 0;
                if ( a1[5] == v28 )
                  return a1[6] < a2[6];
              }
            }
          }
        }
      }
    }
  }
  return result;
}
