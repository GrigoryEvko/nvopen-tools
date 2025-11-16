// Function: sub_867130
// Address: 0x867130
//
__int64 __fastcall sub_867130(__int64 a1, _QWORD *a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  char v6; // r13
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 *i; // rax
  __int64 *v11; // r14
  __int64 *v12; // rax
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r15
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  char v23; // al
  __int64 v24; // rax

  v6 = (char)a4;
  v7 = a1;
  v8 = (unsigned int)dword_4F04C44;
  if ( dword_4F04C44 != -1
    || (a4 = qword_4F04C68, result = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(result + 6) & 2) != 0) )
  {
    result = (__int64)qword_4F04C18;
    if ( !qword_4F04C18 || !qword_4F04C18[2] && (!*((_BYTE *)qword_4F04C18 + 40) || *((_BYTE *)qword_4F04C18 + 42)) )
    {
      if ( a3
        || (result = dword_4F04C64, dword_4F04C64 != -1)
        && (result = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(result + 7) & 1) != 0)
        && (dword_4F04C44 != -1 || (*(_BYTE *)(result + 6) & 6) != 0 || *(_BYTE *)(result + 4) == 12)
        && (result = sub_8782F0(a1), (_DWORD)result) )
      {
        if ( a1 && (*(_WORD *)(a1 + 80) & 0x40FF) == 3 )
        {
          for ( i = *(__int64 **)(a1 + 88); *((_BYTE *)i + 140) == 12; i = (__int64 *)i[20] )
            ;
          v7 = *i;
        }
        v11 = (__int64 *)(sub_85B130(a1, a2, v8, a4, a5) + 664);
        v12 = (__int64 *)*v11;
        if ( *v11 )
        {
          while ( 1 )
          {
            v13 = *((_DWORD *)v12 + 7);
            if ( v12[1] == v7 && dword_4F06650[0] == v13 )
              goto LABEL_18;
            if ( dword_4F06650[0] >= v13 )
            {
              v11 = v12;
              v12 = (__int64 *)*v12;
              if ( v12 )
                continue;
            }
            break;
          }
        }
        if ( a3 )
        {
          v14 = sub_866270(4);
          *(_QWORD *)(v14 + 8) = v7;
          v15 = v14;
          *(_BYTE *)(v14 + 97) = v6;
          goto LABEL_33;
        }
        v16 = *(_BYTE *)(v7 + 80);
        if ( v16 == 7 )
        {
          v20 = sub_866270(1);
          *(_QWORD *)(v20 + 8) = v7;
          v15 = v20;
          *(_DWORD *)(v20 + 16) = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 88) + 128LL) + 36LL);
          if ( dword_4F04C58 != -1
            && (*(_BYTE *)(v7 + 84) & 0x40) == 0
            && *(_DWORD *)(v7 + 40) == *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58) )
          {
            goto LABEL_33;
          }
        }
        else
        {
          if ( v16 != 18 )
          {
            if ( v16 == 8 )
            {
              v24 = sub_866270(3);
              *(_QWORD *)(v24 + 8) = v7;
              v15 = v24;
            }
            else
            {
              v17 = sub_866270(0);
              *(_QWORD *)(v17 + 8) = v7;
              v15 = v17;
              v18 = qword_4F04C68[0];
              v19 = unk_4F04C48;
              if ( dword_4F04C44 >= unk_4F04C48 )
                v19 = dword_4F04C44;
              if ( *(_DWORD *)(v7 + 40) == *(_DWORD *)(qword_4F04C68[0] + 776LL * v19) )
              {
                v18 = 776LL * dword_4F04C64 + qword_4F04C68[0];
                if ( *(_BYTE *)(v18 + 4) == 1 )
                {
                  v18 = *(_QWORD *)(v18 + 624);
                  if ( v18 )
                  {
                    v23 = *(_BYTE *)(v18 + 131);
                    if ( (v23 & 0x40) != 0 )
                      *(_BYTE *)(v18 + 131) = v23 & 0xBF;
                  }
                }
              }
              else
              {
                *(_BYTE *)(v15 + 96) = 1;
              }
              *(_QWORD *)(v15 + 64) = sub_892AE0(v7, &dword_4F04C44, v18);
            }
            goto LABEL_33;
          }
          v21 = sub_866270(2);
          *(_QWORD *)(v21 + 8) = v7;
          v15 = v21;
          v22 = *(_QWORD *)(v7 + 88);
          *(_DWORD *)(v15 + 16) = *(_DWORD *)(v22 + 120);
          if ( (*(_BYTE *)(v22 + 42) & 4) == 0 )
          {
LABEL_33:
            *(_QWORD *)(v15 + 20) = *a2;
            *(_DWORD *)(v15 + 28) = dword_4F06650[0];
            if ( *v11 )
              *(_QWORD *)v15 = *v11;
            *v11 = v15;
LABEL_18:
            result = (__int64)qword_4F04C18;
            if ( qword_4F04C18 )
              *((_BYTE *)qword_4F04C18 + 52) = 1;
            return result;
          }
        }
        *(_BYTE *)(v15 + 96) = 1;
        goto LABEL_33;
      }
    }
  }
  return result;
}
