// Function: sub_133F890
// Address: 0x133f890
//
void __fastcall sub_133F890(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // rdx
  int v3; // eax
  _QWORD *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  unsigned __int64 v7; // r11
  unsigned int v9; // r10d
  _QWORD *v10; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // r8
  __int64 v13; // rsi
  __int64 v14; // r9
  _QWORD *v15; // rcx
  unsigned __int64 v16; // rsi
  int v17; // esi
  __int64 v18; // rsi

  a2[5] = 0;
  a2[6] = 0;
  a2[7] = 0;
  v2 = (_QWORD *)*a1;
  if ( *a1 )
  {
    v3 = (a2[4] > v2[4]) - (a2[4] < v2[4]);
    if ( a2[4] > v2[4] == a2[4] < v2[4] )
      v3 = (a2[1] > v2[1]) - (a2[1] < v2[1]);
    if ( v3 == -1 )
    {
      a2[7] = v2;
      v2[5] = a2;
      *a1 = a2;
      a1[1] = 0;
      return;
    }
    ++a1[1];
    a2[6] = v2[6];
    v4 = (_QWORD *)*a1;
    v5 = *(_QWORD *)(*a1 + 48LL);
    v6 = *a1 + 40LL;
    if ( v5 )
    {
      *(_QWORD *)(v5 + 40) = a2;
      v4 = (_QWORD *)*a1;
      v6 = *a1 + 40LL;
    }
    a2[5] = v4;
    *(_QWORD *)(v6 + 8) = a2;
  }
  else
  {
    *a1 = a2;
  }
  v7 = a1[1];
  if ( v7 > 1 )
  {
    if ( !_BitScanForward64(&v7, v7 - 1) )
      LODWORD(v7) = -1;
    if ( (_DWORD)v7 )
    {
      v9 = 0;
      v10 = *(_QWORD **)(*a1 + 48LL);
      if ( v10 )
      {
        v11 = (_QWORD *)v10[6];
        v12 = v10 + 5;
        if ( v11 )
        {
          while ( 1 )
          {
            v14 = v11[6];
            v12[1] = 0;
            v15 = v11 + 5;
            v10[5] = 0;
            v11[6] = 0;
            v16 = v11[4];
            v11[5] = 0;
            v17 = (v10[4] > v16) - (v10[4] < v16);
            if ( !v17 )
              v17 = (v10[1] > v11[1]) - (v10[1] < v11[1]);
            if ( v17 == -1 )
            {
              *v15 = v10;
              v18 = v12[2];
              v11[6] = v18;
              if ( v18 )
                *(_QWORD *)(v18 + 40) = v11;
              v15 = v12;
              v12[2] = v11;
              v11 = v10;
              v12[1] = v14;
              if ( !v14 )
                goto LABEL_28;
            }
            else
            {
              v10[5] = v11;
              v13 = v11[7];
              v12[1] = v13;
              if ( v13 )
                *(_QWORD *)(v13 + 40) = v10;
              v11[7] = v10;
              v11[6] = v14;
              if ( !v14 )
              {
LABEL_28:
                *(_QWORD *)(*a1 + 48LL) = v11;
                *v15 = *a1;
                return;
              }
            }
            *(_QWORD *)(v14 + 40) = v11;
            ++v9;
            *(_QWORD *)(*a1 + 48LL) = v11;
            *v15 = *a1;
            if ( v9 < (unsigned int)v7 )
            {
              v10 = *(_QWORD **)(*a1 + 48LL);
              if ( v10 )
              {
                v11 = (_QWORD *)v10[6];
                v12 = v10 + 5;
                if ( v11 )
                  continue;
              }
            }
            return;
          }
        }
      }
    }
  }
}
