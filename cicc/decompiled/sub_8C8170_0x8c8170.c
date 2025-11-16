// Function: sub_8C8170
// Address: 0x8c8170
//
__int64 *__fastcall sub_8C8170(__int64 a1, __int64 a2)
{
  __int64 *v2; // r15
  __int64 v4; // rbx
  char v5; // r13
  __int64 v6; // r14
  char v7; // dl
  __int64 v8; // rax
  __int64 v10; // r9
  __int64 v11; // r10
  __int64 v12; // rsi
  __int64 i; // rdi
  int v14; // eax
  int v15; // eax
  __int64 **v16; // r9
  _BOOL4 v17; // eax
  _BOOL4 v18; // eax
  __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int64 **v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+18h] [rbp-38h]

  v2 = 0;
  v4 = a2;
  v23 = sub_880F80(a1);
  v22 = *(_QWORD *)(a1 + 88);
  while ( v4 )
  {
    if ( *(_DWORD *)(v4 + 40) == -1 )
      goto LABEL_3;
    if ( v23 == sub_880F80(v4) )
      goto LABEL_3;
    v5 = *(_BYTE *)(v4 + 80);
    v6 = v4;
    if ( v5 == 17 )
    {
      v6 = *(_QWORD *)(v4 + 88);
      if ( !v6 )
        goto LABEL_3;
    }
    if ( (unsigned int)sub_8C7F70(v6, a1) )
      goto LABEL_11;
    while ( v5 == 17 )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        break;
      if ( (unsigned int)sub_8C7F70(v6, a1) )
      {
LABEL_11:
        if ( (unsigned int)sub_8C6B40(v6) )
        {
          if ( !v2 )
          {
            v7 = *(_BYTE *)(v6 + 80);
            switch ( v7 )
            {
              case 3:
                if ( !*(_BYTE *)(v4 + 104) )
                  goto LABEL_21;
                continue;
              case 4:
              case 5:
              case 6:
              case 20:
                continue;
              case 10:
              case 11:
              case 15:
                v10 = *(_QWORD *)(v6 + 88);
                if ( v7 == 15 )
                  v10 = *(_QWORD *)(v10 + 8);
                if ( v22 == v10
                  || (*(_BYTE *)(v10 + 195) & 1) != 0
                  || *(_BYTE *)(v22 + 174) == 5
                  && *(_BYTE *)(v10 + 174) == 5
                  && *(_BYTE *)(v22 + 176) != *(_BYTE *)(v10 + 176) )
                {
                  continue;
                }
                v11 = *(_QWORD *)(v10 + 152);
                if ( ((*(_BYTE *)(v22 + 88) ^ *(_BYTE *)(v10 + 88)) & 0x70) != 0 )
                {
                  v12 = *(_QWORD *)(v10 + 152);
                  if ( *(_BYTE *)(v11 + 140) == 12 )
                  {
                    do
                      v12 = *(_QWORD *)(v12 + 160);
                    while ( *(_BYTE *)(v12 + 140) == 12 );
                  }
                  for ( i = *(_QWORD *)(v22 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
                    ;
                  v19 = *(_QWORD *)(v10 + 152);
                  v20 = v10;
                  v14 = sub_8D73A0(i, v12);
                  v10 = v20;
                  v11 = v19;
                  if ( !v14 )
                    continue;
                }
                v21 = (__int64 **)v10;
                v15 = sub_8DE890(*(_QWORD *)(v22 + 152), v11, 260, 0);
                v16 = v21;
                if ( v15
                  || (v17 = sub_72F8B0((__int64 **)v22), v16 = v21, v17) && (v18 = sub_72F8B0(v21), v16 = v21, v18) )
                {
                  v2 = *v16;
                }
                else if ( (*(_BYTE *)(v22 + 88) & 0x70) == 0x30 && ((_BYTE)v16[11] & 0x70) == 0x30 )
                {
                  if ( (*(_BYTE *)(v22 + 193) & 0x10) == 0 )
                  {
                    if ( (*((_BYTE *)v16 + 193) & 0x10) == 0 )
                      goto LABEL_21;
                    goto LABEL_46;
                  }
                  if ( !*(_BYTE *)(v22 + 174) && dword_4F077C4 != 2 && !*(_WORD *)(v22 + 176) )
                    goto LABEL_21;
                  if ( (*((_BYTE *)v16 + 193) & 0x10) != 0 )
                  {
LABEL_46:
                    if ( !*((_BYTE *)v16 + 174) && dword_4F077C4 != 2 && !*((_WORD *)v16 + 88) )
                      goto LABEL_21;
                  }
                }
                break;
              default:
LABEL_21:
                sub_8C6700((__int64 *)v22, (unsigned int *)(v6 + 48), 0x42Au, 0x425u);
                continue;
            }
          }
        }
        else
        {
          v8 = sub_87D520(v6);
          if ( v8 && (*(_BYTE *)(v8 - 8) & 2) == 0 )
            *(_BYTE *)(v8 + 90) |= 8u;
        }
      }
    }
LABEL_3:
    v4 = *(_QWORD *)(v4 + 8);
  }
  return v2;
}
