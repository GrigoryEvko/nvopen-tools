// Function: sub_D955C0
// Address: 0xd955c0
//
void __fastcall sub_D955C0(__int64 a1, __int64 a2)
{
  __int16 v2; // cx
  __int64 v4; // r12
  __int64 v5; // r12
  const char *v6; // rsi
  __int64 v7; // rdi
  unsigned int v8; // r12d
  __int64 v9; // rax
  int v10; // r13d
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r13
  const char *v15; // rsi
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r12
  unsigned __int8 *v20; // rdi
  __int16 v21; // ax
  char v22; // r13
  size_t v23; // r12
  __int64 *v24; // r15
  __int64 v25; // rsi
  __int64 v26; // r9
  _QWORD *v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rsi
  char *v30; // rax
  char *v31; // rcx
  unsigned int v32; // eax
  unsigned int v33; // eax
  unsigned int v34; // edx
  __int64 v35; // r10
  __int16 v36; // ax
  __int64 v37; // [rsp+8h] [rbp-48h]
  char *s; // [rsp+10h] [rbp-40h]
  __int64 *v39; // [rsp+18h] [rbp-38h]

  v2 = *(_WORD *)(a1 + 24);
  switch ( v2 )
  {
    case 0:
      v20 = *(unsigned __int8 **)(a1 + 32);
      goto LABEL_14;
    case 1:
      v6 = "vscale";
      goto LABEL_16;
    case 2:
      v14 = *(_QWORD *)(a1 + 32);
      v15 = "(trunc ";
      goto LABEL_12;
    case 3:
      v14 = *(_QWORD *)(a1 + 32);
      v15 = "(zext ";
      goto LABEL_12;
    case 4:
      v14 = *(_QWORD *)(a1 + 32);
      v15 = "(sext ";
      goto LABEL_12;
    case 5:
    case 6:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
      switch ( v2 )
      {
        case 5:
          s = " + ";
          goto LABEL_31;
        case 6:
          s = " * ";
          goto LABEL_31;
        case 9:
          s = " umax ";
          goto LABEL_31;
        case 10:
          s = " smax ";
          goto LABEL_31;
        case 11:
          s = " umin ";
          goto LABEL_31;
        case 12:
          s = " smin ";
          goto LABEL_31;
        case 13:
          s = " umin_seq ";
LABEL_31:
          v22 = 1;
          sub_904010(a2, "(");
          v23 = strlen(s);
          v24 = *(__int64 **)(a1 + 32);
          v39 = &v24[*(_QWORD *)(a1 + 40)];
          while ( v39 != v24 )
          {
            v26 = *v24;
            if ( v22 )
            {
              v25 = a2;
              v22 = 0;
            }
            else
            {
              v27 = *(_QWORD **)(a2 + 32);
              if ( v23 <= *(_QWORD *)(a2 + 24) - (_QWORD)v27 )
              {
                if ( (unsigned int)v23 >= 8 )
                {
                  *v27 = *(_QWORD *)s;
                  *(_QWORD *)((char *)v27 + (unsigned int)v23 - 8) = *(_QWORD *)&s[(unsigned int)v23 - 8];
                  v29 = (unsigned __int64)(v27 + 1) & 0xFFFFFFFFFFFFFFF8LL;
                  v30 = (char *)v27 - v29;
                  v31 = (char *)(s - v30);
                  v32 = (v23 + (_DWORD)v30) & 0xFFFFFFF8;
                  if ( v32 >= 8 )
                  {
                    v33 = v32 & 0xFFFFFFF8;
                    v34 = 0;
                    do
                    {
                      v35 = v34;
                      v34 += 8;
                      *(_QWORD *)(v29 + v35) = *(_QWORD *)&v31[v35];
                    }
                    while ( v34 < v33 );
                  }
                }
                else if ( (v23 & 4) != 0 )
                {
                  *(_DWORD *)v27 = *(_DWORD *)s;
                  *(_DWORD *)((char *)v27 + (unsigned int)v23 - 4) = *(_DWORD *)&s[(unsigned int)v23 - 4];
                }
                else if ( (_DWORD)v23 )
                {
                  *(_BYTE *)v27 = *s;
                  if ( (v23 & 2) != 0 )
                    *(_WORD *)((char *)v27 + (unsigned int)v23 - 2) = *(_WORD *)&s[(unsigned int)v23 - 2];
                }
                *(_QWORD *)(a2 + 32) += v23;
                v25 = a2;
              }
              else
              {
                v37 = *v24;
                v28 = sub_CB6200(a2, (unsigned __int8 *)s, v23);
                v26 = v37;
                v25 = v28;
              }
            }
            ++v24;
            sub_D955C0(v26, v25);
          }
          sub_904010(a2, ")");
          if ( (unsigned int)*(unsigned __int16 *)(a1 + 24) - 5 <= 1 )
          {
            v36 = *(_WORD *)(a1 + 28);
            if ( (v36 & 2) != 0 )
            {
              sub_904010(a2, "<nuw>");
              v36 = *(_WORD *)(a1 + 28);
            }
            v6 = "<nsw>";
            if ( (v36 & 4) != 0 )
              goto LABEL_16;
          }
          return;
        default:
          goto LABEL_61;
      }
    case 7:
      v4 = sub_904010(a2, "(");
      sub_D955C0(*(_QWORD *)(a1 + 32), v4);
      v5 = sub_904010(v4, " /u ");
      sub_D955C0(*(_QWORD *)(a1 + 40), v5);
      v6 = ")";
      v7 = v5;
      goto LABEL_4;
    case 8:
      v8 = 1;
      v9 = sub_904010(a2, "{");
      sub_D955C0(**(_QWORD **)(a1 + 32), v9);
      v10 = *(_QWORD *)(a1 + 40);
      if ( v10 != 1 )
      {
        do
        {
          v13 = *(_QWORD *)(a2 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v13) > 2 )
          {
            *(_BYTE *)(v13 + 2) = 44;
            v11 = a2;
            *(_WORD *)v13 = 11052;
            *(_QWORD *)(a2 + 32) += 3LL;
          }
          else
          {
            v11 = sub_CB6200(a2, ",+,", 3u);
          }
          v12 = v8++;
          sub_D955C0(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v12), v11);
        }
        while ( v10 != v8 );
      }
      sub_904010(a2, "}<");
      v21 = *(_WORD *)(a1 + 28);
      if ( (v21 & 2) != 0 )
      {
        sub_904010(a2, "nuw><");
        v21 = *(_WORD *)(a1 + 28);
      }
      if ( (v21 & 4) != 0 )
      {
        sub_904010(a2, "nsw><");
        v21 = *(_WORD *)(a1 + 28);
      }
      if ( (v21 & 1) != 0 && (v21 & 6) == 0 )
        sub_904010(a2, "nw><");
      sub_A5BF40(**(unsigned __int8 ***)(*(_QWORD *)(a1 + 48) + 32LL), a2, 0, 0);
      v6 = ">";
      goto LABEL_16;
    case 14:
      v14 = *(_QWORD *)(a1 + 32);
      v15 = "(ptrtoint ";
LABEL_12:
      v16 = sub_904010(a2, v15);
      v17 = sub_D95540(v14);
      sub_A587F0(v17, v16, 0, 0);
      v18 = sub_904010(v16, " ");
      sub_D955C0(v14, v18);
      v19 = sub_904010(v18, " to ");
      sub_A587F0(*(_QWORD *)(a1 + 40), v19, 0, 0);
      sub_904010(v19, ")");
      break;
    case 15:
      v20 = *(unsigned __int8 **)(a1 - 8);
LABEL_14:
      sub_A5BF40(v20, a2, 0, 0);
      break;
    case 16:
      v6 = "***COULDNOTCOMPUTE***";
LABEL_16:
      v7 = a2;
LABEL_4:
      sub_904010(v7, v6);
      break;
    default:
LABEL_61:
      BUG();
  }
}
