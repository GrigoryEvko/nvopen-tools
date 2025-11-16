// Function: sub_8E5310
// Address: 0x8e5310
//
__int64 __fastcall sub_8E5310(__int128 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v8; // rdx
  __int64 result; // rax
  unsigned __int64 v10; // r14
  unsigned int i; // edx
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int8 v17; // dl
  unsigned __int8 v18; // al
  __int64 v19; // rcx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  _QWORD *v25; // r15
  __int64 v26; // [rsp+18h] [rbp-58h] BYREF
  __int128 v27; // [rsp+20h] [rbp-50h]
  unsigned __int64 v28; // [rsp+30h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 56);
  if ( v5 == *((_QWORD *)&a1 + 1) )
    return a1;
  v6 = *((_QWORD *)&a1 + 1);
  if ( !v5
    || !*((_QWORD *)&a1 + 1)
    || (a5 = dword_4F07588) == 0
    || (v8 = *(_QWORD *)(v5 + 32), *(_QWORD *)(*((_QWORD *)&a1 + 1) + 32LL) != v8)
    || (result = a1, !v8) )
  {
    v27 = a1;
    v28 = a2;
    v10 = (a2 >> 3) + 31 * (31 * (((unsigned __int64)a1 >> 3) + 527) + (*((_QWORD *)&a1 + 1) >> 3));
    for ( i = v10 & *(_DWORD *)(qword_4F60598 + 8); ; i = *(_DWORD *)(qword_4F60598 + 8) & (i + 1) )
    {
      v12 = *(_QWORD *)qword_4F60598 + 32LL * i;
      if ( (_QWORD)a1 == *(_QWORD *)v12 )
      {
        if ( __PAIR128__(a2, *((unsigned __int64 *)&a1 + 1)) == *(_OWORD *)(v12 + 8) )
        {
          result = *(_QWORD *)(v12 + 24);
          v26 = result;
          if ( result )
            return result;
LABEL_17:
          v13 = **(_QWORD **)(*((_QWORD *)&a1 + 1) + 168LL);
          if ( !v13 )
            return 0;
          while ( 1 )
          {
LABEL_25:
            v15 = *(_QWORD *)(v13 + 40);
            v16 = *(_QWORD *)(a1 + 40);
            if ( v15 != v16 )
            {
              if ( !v16 )
                goto LABEL_24;
              if ( !v15 )
                goto LABEL_24;
              v6 = dword_4F07588;
              if ( !dword_4F07588 )
                goto LABEL_24;
              v14 = *(_QWORD *)(v15 + 32);
              if ( *(_QWORD *)(v16 + 32) != v14 || !v14 )
                goto LABEL_24;
            }
            v17 = *(_BYTE *)(v13 + 96);
            v18 = *(_BYTE *)(a1 + 96);
            v19 = v17 & 2;
            if ( (v18 & 2) != 0 )
              break;
            if ( (_BYTE)v19 )
              goto LABEL_24;
            if ( (v18 & 1) == 0 )
            {
              if ( ((v17 | v18) & 4) == 0 )
                goto LABEL_28;
              if ( a2 )
              {
                if ( (*(_BYTE *)(v13 + 96) & 1) == 0 )
                {
                  v20 = *(_QWORD **)(*(_QWORD *)(v13 + 112) + 8LL);
                  do
                  {
                    v21 = v20;
                    v20 = (_QWORD *)*v20;
                  }
                  while ( v20[2] != v13 );
                  if ( v21[2] == a2 )
                    goto LABEL_28;
                }
              }
              else
              {
                v24 = *(_QWORD *)(v13 + 112);
                *((_QWORD *)&a1 + 1) = *(_QWORD *)(v24 + 16);
                v25 = *(_QWORD **)(v24 + 8);
                if ( v25 != **((_QWORD ***)&a1 + 1) )
                {
                  while ( !(unsigned int)sub_5ED650(
                                           v25,
                                           *((_QWORD **)&a1 + 1),
                                           *(_QWORD **)(*(_QWORD *)(a1 + 112) + 8LL),
                                           *(_QWORD **)(*(_QWORD *)(a1 + 112) + 16LL)) )
                  {
                    v25 = (_QWORD *)*v25;
                    if ( **((_QWORD ***)&a1 + 1) == v25 )
                      goto LABEL_24;
                  }
LABEL_28:
                  v26 = v13;
                  sub_8E5140(qword_4F60598, &v26, v10, v19, v6, a5, v27, v28);
                  return v26;
                }
              }
              goto LABEL_24;
            }
            if ( (v17 & 1) == 0 )
            {
              v22 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v13 + 112) + 16LL) + 8LL) + 16LL);
              v23 = *(_QWORD *)(a1 + 56);
              v19 = *(_QWORD *)(v22 + 40);
              if ( v19 == v23
                || v23 && v19 && dword_4F07588 && (v19 = *(_QWORD *)(v19 + 32), *(_QWORD *)(v23 + 32) == v19) && v19 )
              {
                if ( ((a2 != 0) & (v17 >> 2)) == 0 || v22 == a2 )
                  goto LABEL_28;
              }
              goto LABEL_24;
            }
            if ( (*(_BYTE *)(v13 + 96) & 4) == 0 || !a2 )
              goto LABEL_28;
            v13 = *(_QWORD *)v13;
            if ( !v13 )
              return 0;
          }
          if ( (_BYTE)v19 )
            goto LABEL_28;
LABEL_24:
          v13 = *(_QWORD *)v13;
          if ( !v13 )
            return 0;
          goto LABEL_25;
        }
      }
      else if ( !*(_QWORD *)v12 && !*(_QWORD *)(v12 + 8) && !*(_QWORD *)(v12 + 16) )
      {
        v26 = 0;
        goto LABEL_17;
      }
    }
  }
  return result;
}
