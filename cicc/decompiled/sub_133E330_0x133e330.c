// Function: sub_133E330
// Address: 0x133e330
//
void __fastcall sub_133E330(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // rdx
  unsigned __int64 v3; // r8
  int v4; // eax
  _QWORD *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // rbx
  unsigned int v10; // r10d
  __int64 v11; // rsi
  _QWORD *v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // r9
  _QWORD *v15; // rcx
  __int64 v16; // rsi
  int v17; // esi
  __int64 v18; // rsi

  a2[5] = 0;
  a2[6] = 0;
  a2[7] = 0;
  v2 = (_QWORD *)*a1;
  if ( *a1 )
  {
    v3 = a2[2] & 0xFFFLL;
    v4 = (v3 > (v2[2] & 0xFFFuLL)) - (v3 < (v2[2] & 0xFFFuLL));
    if ( v3 > (v2[2] & 0xFFFuLL) == v3 < (v2[2] & 0xFFFuLL) )
      v4 = (a2 > v2) - (a2 < v2);
    if ( v4 == -1 )
    {
      a2[7] = v2;
      v2[5] = a2;
      *a1 = a2;
      a1[1] = 0;
      return;
    }
    ++a1[1];
    a2[6] = v2[6];
    v5 = (_QWORD *)*a1;
    v6 = *(_QWORD *)(*a1 + 48LL);
    v7 = *a1 + 40LL;
    if ( v6 )
    {
      *(_QWORD *)(v6 + 40) = a2;
      v5 = (_QWORD *)*a1;
      v7 = *a1 + 40LL;
    }
    a2[5] = v5;
    *(_QWORD *)(v7 + 8) = a2;
  }
  else
  {
    *a1 = a2;
  }
  v8 = a1[1];
  if ( v8 > 1 )
  {
    if ( !_BitScanForward64(&v8, v8 - 1) )
      LODWORD(v8) = -1;
    if ( (_DWORD)v8 )
    {
      v10 = 0;
      do
      {
        v12 = *(_QWORD **)(*a1 + 48LL);
        if ( !v12 )
          break;
        v13 = (_QWORD *)v12[6];
        if ( !v13 )
          break;
        v14 = v13[6];
        v12[6] = 0;
        v15 = v13 + 5;
        v12[5] = 0;
        v13[6] = 0;
        v16 = v13[2];
        v13[5] = 0;
        v17 = ((v12[2] & 0xFFFuLL) > ((unsigned __int16)v16 & 0xFFFu))
            - ((v12[2] & 0xFFFuLL) < ((unsigned __int16)v16 & 0xFFFu));
        if ( !v17 )
          v17 = (v12 > v13) - (v12 < v13);
        if ( v17 == -1 )
        {
          *v15 = v12;
          v18 = v12[7];
          v13[6] = v18;
          if ( v18 )
            *(_QWORD *)(v18 + 40) = v13;
          v15 = v12 + 5;
          v12[7] = v13;
          v13 = v12;
          v12[6] = v14;
          if ( !v14 )
            goto LABEL_26;
        }
        else
        {
          v12[5] = v13;
          v11 = v13[7];
          v12[6] = v11;
          if ( v11 )
            *(_QWORD *)(v11 + 40) = v12;
          v13[7] = v12;
          v13[6] = v14;
          if ( !v14 )
          {
LABEL_26:
            *(_QWORD *)(*a1 + 48LL) = v13;
            *v15 = *a1;
            return;
          }
        }
        *(_QWORD *)(v14 + 40) = v13;
        ++v10;
        *(_QWORD *)(*a1 + 48LL) = v13;
        *v15 = *a1;
      }
      while ( v10 < (unsigned int)v8 );
    }
  }
}
