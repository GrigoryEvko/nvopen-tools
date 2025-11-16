// Function: sub_2B2BA00
// Address: 0x2b2ba00
//
__int64 __fastcall sub_2B2BA00(__int64 a1, __int64 a2, __int64 a3, __int64 i, __int64 a5)
{
  __int64 v5; // r12
  unsigned int v6; // r14d
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned int v11; // eax
  _BYTE *v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // edi
  unsigned int v18; // eax
  __int64 v19; // r9
  unsigned int v20; // r10d

  v5 = *(_QWORD *)(a2 + 16);
  if ( v5 )
  {
    if ( *(_QWORD *)(v5 + 8) )
      goto LABEL_9;
    if ( a3 )
    {
      if ( (*(_BYTE *)(a3 + 8) & 1) != 0 )
      {
        a5 = a3 + 16;
        v17 = 3;
        v16 = a3 + 48;
      }
      else
      {
        a5 = *(_QWORD *)(a3 + 16);
        v15 = *(unsigned int *)(a3 + 24);
        v16 = a5 + 8 * v15;
        if ( !(_DWORD)v15 )
          goto LABEL_9;
        v17 = v15 - 1;
      }
      v18 = v17 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      i = a5 + 8LL * v18;
      v19 = *(_QWORD *)i;
      if ( a2 != *(_QWORD *)i )
      {
        for ( i = 1; ; i = v20 )
        {
          if ( v19 == -4096 )
            goto LABEL_9;
          v20 = i + 1;
          v18 = v17 & (i + v18);
          i = a5 + 8LL * v18;
          v19 = *(_QWORD *)i;
          if ( a2 == *(_QWORD *)i )
            break;
        }
      }
      if ( i == v16 )
      {
        do
        {
LABEL_9:
          v12 = *(_BYTE **)(v5 + 24);
          if ( (*(_BYTE *)(a1 + 88) & 1) != 0 )
          {
            v9 = a1 + 96;
            v10 = 3;
          }
          else
          {
            v10 = *(unsigned int *)(a1 + 104);
            v9 = *(_QWORD *)(a1 + 96);
            if ( !(_DWORD)v10 )
              goto LABEL_14;
            v10 = (unsigned int)(v10 - 1);
          }
          v11 = v10 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          i = *(_QWORD *)(v9 + 72LL * v11);
          if ( (_BYTE *)i == v12 )
            goto LABEL_8;
          LODWORD(a5) = 1;
          while ( i != -4096 )
          {
            v11 = v10 & (a5 + v11);
            i = *(_QWORD *)(v9 + 72LL * v11);
            if ( v12 == (_BYTE *)i )
              goto LABEL_8;
            LODWORD(a5) = a5 + 1;
          }
LABEL_14:
          v6 = sub_2B15E10(*(char **)(v5 + 24), (__int64)v12, v10, i, a5);
          if ( !(_BYTE)v6 )
          {
            if ( *v12 != 90 )
              return v6;
            if ( *(_BYTE *)(a1 + 796) )
            {
              v13 = *(_QWORD **)(a1 + 776);
              v14 = &v13[*(unsigned int *)(a1 + 788)];
              if ( v13 == v14 )
                return v6;
              while ( v12 != (_BYTE *)*v13 )
              {
                if ( v14 == ++v13 )
                  return v6;
              }
            }
            else if ( !sub_C8CA60(a1 + 768, (__int64)v12) )
            {
              return v6;
            }
          }
LABEL_8:
          v5 = *(_QWORD *)(v5 + 8);
        }
        while ( v5 );
      }
    }
  }
  return 1;
}
