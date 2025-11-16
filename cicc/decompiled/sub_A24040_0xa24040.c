// Function: sub_A24040
// Address: 0xa24040
//
void __fastcall sub_A24040(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, unsigned int **a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // rax
  _BYTE *v15; // rsi
  __int64 v16; // rax
  char v17; // al
  unsigned __int64 v18; // rdx
  __int64 *v19; // r15
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 *v22; // rbx
  int v23; // esi
  __int64 v24; // r8
  __int64 v25; // r10
  __int64 v26; // r9
  int v27; // esi
  unsigned int v28; // ecx
  __int64 v29; // rdx
  __int64 v30; // r11
  unsigned int v31; // eax
  unsigned int v32; // eax
  int v33; // edx
  _BOOL8 v34; // rsi
  unsigned int v35; // r15d
  unsigned int v36; // ecx
  int v37; // r10d
  __int64 v38; // [rsp+10h] [rbp-70h]
  _QWORD *v40; // [rsp+20h] [rbp-60h]
  __int64 v41; // [rsp+28h] [rbp-58h]
  __int64 v42; // [rsp+28h] [rbp-58h]
  __int64 *v43; // [rsp+30h] [rbp-50h]
  __int64 v44; // [rsp+38h] [rbp-48h]
  unsigned int v45; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v46; // [rsp+44h] [rbp-3Ch] BYREF
  __int64 v47[7]; // [rsp+48h] [rbp-38h] BYREF

  if ( a3 )
  {
    v6 = a2;
    v45 = 0;
    v46 = 0;
    v43 = &a2[a3];
    v38 = (__int64)(a1 + 3);
    if ( v43 != a2 )
    {
      do
      {
        v10 = *v6;
        if ( a6 )
        {
          v11 = *a1;
          v12 = *(_QWORD *)(*a1 + 32);
          v44 = *(_QWORD *)(*(_QWORD *)(*a1 + 24) + 8LL);
          if ( v12 )
          {
            v41 = *a1;
            v40 = *(_QWORD **)(*a1 + 32);
            v13 = sub_CB7440(v12);
            v11 = v41;
            if ( v13 )
            {
              if ( !(unsigned __int8)sub_CB7440(v40) )
                BUG();
              v14 = (*(__int64 (__fastcall **)(_QWORD *))(*v40 + 80LL))(v40);
              v11 = v41;
              v44 += v14 + v40[4] - v40[2];
            }
          }
          v15 = *(_BYTE **)(a6 + 8);
          v16 = *(unsigned int *)(v11 + 48) + 8 * v44;
          v47[0] = v16;
          if ( v15 == *(_BYTE **)(a6 + 16) )
          {
            sub_A235E0(a6, v15, v47);
          }
          else
          {
            if ( v15 )
            {
              *(_QWORD *)v15 = v16;
              v15 = *(_BYTE **)(a6 + 8);
            }
            *(_QWORD *)(a6 + 8) = v15 + 8;
          }
        }
        v17 = *(_BYTE *)v10;
        if ( (unsigned __int8)(*(_BYTE *)v10 - 5) > 0x1Fu )
        {
          if ( v17 == 4 )
          {
            v18 = *(unsigned int *)(v10 + 144);
            if ( v18 > *(unsigned int *)(a4 + 12) )
            {
              sub_C8D5F0(a4, a4 + 16, v18, 8);
              v18 = *(unsigned int *)(v10 + 144);
            }
            v19 = *(__int64 **)(v10 + 136);
            if ( v19 != &v19[v18] )
            {
              v20 = *(unsigned int *)(a4 + 8);
              v21 = a4;
              v22 = &v19[v18];
              do
              {
                v23 = *((_DWORD *)a1 + 76);
                v24 = *v19;
                v25 = 0xFFFFFFFFLL;
                v26 = a1[36];
                if ( v23 )
                {
                  v27 = v23 - 1;
                  v28 = v27 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
                  v29 = v26 + 16LL * v28;
                  v30 = *(_QWORD *)v29;
                  if ( v24 == *(_QWORD *)v29 )
                  {
LABEL_21:
                    v25 = (unsigned int)(*(_DWORD *)(v29 + 12) - 1);
                  }
                  else
                  {
                    v33 = 1;
                    while ( v30 != -4096 )
                    {
                      v37 = v33 + 1;
                      v28 = v27 & (v33 + v28);
                      v29 = v26 + 16LL * v28;
                      v30 = *(_QWORD *)v29;
                      if ( v24 == *(_QWORD *)v29 )
                        goto LABEL_21;
                      v33 = v37;
                    }
                    v25 = 0xFFFFFFFFLL;
                  }
                }
                if ( v20 + 1 > (unsigned __int64)*(unsigned int *)(v21 + 12) )
                {
                  v42 = v25;
                  sub_C8D5F0(v21, v21 + 16, v20 + 1, 8);
                  v25 = v42;
                  v20 = *(unsigned int *)(v21 + 8);
                }
                ++v19;
                *(_QWORD *)(*(_QWORD *)v21 + 8 * v20) = v25;
                v20 = (unsigned int)(*(_DWORD *)(v21 + 8) + 1);
                *(_DWORD *)(v21 + 8) = v20;
              }
              while ( v22 != v19 );
              a4 = v21;
            }
            sub_A1BFB0(*a1, 0x2Eu, a4, 0);
            *(_DWORD *)(a4 + 8) = 0;
          }
          else
          {
            v31 = sub_A172F0(v38, *(_QWORD *)(*(_QWORD *)(v10 + 136) + 8LL));
            sub_A188E0(a4, v31);
            v32 = sub_A3F3B0(v38);
            sub_A188E0(a4, v32);
            sub_A1BFB0(*a1, 2u, a4, 0);
            *(_DWORD *)(a4 + 8) = 0;
          }
        }
        else
        {
          switch ( v17 )
          {
            case 6:
              if ( a5 )
                sub_A239F0(a1, v10, a4, *a5 + 1);
              else
                sub_A239F0(a1, v10, a4, &v45);
              break;
            case 7:
              if ( a5 )
                sub_A1C370(a1, v10, a4, (*a5)[2]);
              else
                sub_A1C370(a1, v10, a4, 0);
              break;
            case 8:
              if ( a5 )
                sub_A1C4C0(a1, v10, a4, (*a5)[3]);
              else
                sub_A1C4C0(a1, v10, a4, 0);
              break;
            case 9:
              if ( a5 )
                sub_A23E10((__int64)a1, v10, a4, *a5 + 4);
              else
                sub_A23E10((__int64)a1, v10, a4, &v46);
              break;
            case 10:
              if ( a5 )
                sub_A1C690(a1, v10, a4, (*a5)[5]);
              else
                sub_A1C690(a1, v10, a4, 0);
              break;
            case 11:
              if ( a5 )
                sub_A1C850(a1, v10, a4, (*a5)[6]);
              else
                sub_A1C850(a1, v10, a4, 0);
              break;
            case 12:
              if ( a5 )
                sub_A1C9F0(a1, v10, a4, (*a5)[7]);
              else
                sub_A1C9F0(a1, v10, a4, 0);
              break;
            case 13:
              if ( a5 )
                sub_A1CBD0(a1, (_BYTE *)v10, a4, (*a5)[8]);
              else
                sub_A1CBD0(a1, (_BYTE *)v10, a4, 0);
              break;
            case 14:
              if ( a5 )
                sub_A1CE20(a1, (_BYTE *)v10, a4, (*a5)[9]);
              else
                sub_A1CE20(a1, (_BYTE *)v10, a4, 0);
              break;
            case 15:
              if ( a5 )
                sub_A1D140(a1, v10, a4, (*a5)[10]);
              else
                sub_A1D140(a1, v10, a4, 0);
              break;
            case 16:
              if ( a5 )
                sub_A1D300(a1, v10, a4, (*a5)[11]);
              else
                sub_A1D300(a1, v10, a4, 0);
              break;
            case 17:
              if ( a5 )
                sub_A1D4A0(a1, v10, a4, (*a5)[12]);
              else
                sub_A1D4A0(a1, v10, a4, 0);
              break;
            case 18:
              if ( a5 )
                sub_A1D710(a1, (_BYTE *)v10, a4, (*a5)[13]);
              else
                sub_A1D710(a1, (_BYTE *)v10, a4, 0);
              break;
            case 19:
              if ( a5 )
                sub_A1DAA0(a1, (_BYTE *)v10, a4, (*a5)[14]);
              else
                sub_A1DAA0(a1, (_BYTE *)v10, a4, 0);
              break;
            case 20:
              if ( a5 )
                sub_A1DCA0(a1, (_BYTE *)v10, a4, (*a5)[15]);
              else
                sub_A1DCA0(a1, (_BYTE *)v10, a4, 0);
              break;
            case 21:
              if ( a5 )
                sub_A1DEA0(a1, v10, a4, (*a5)[16]);
              else
                sub_A1DEA0(a1, v10, a4, 0);
              break;
            case 22:
              if ( a5 )
                sub_A1E080((__int64)a1, v10, a4, (*a5)[17]);
              else
                sub_A1E080((__int64)a1, v10, a4, 0);
              break;
            case 23:
              if ( a5 )
                sub_A1E2F0(a1, v10, a4, (*a5)[18]);
              else
                sub_A1E2F0(a1, v10, a4, 0);
              break;
            case 24:
              if ( a5 )
                sub_A1E500(a1, v10, a4, (*a5)[19]);
              else
                sub_A1E500(a1, v10, a4, 0);
              break;
            case 25:
              if ( a5 )
                sub_A1E6D0(a1, v10, a4, (*a5)[20]);
              else
                sub_A1E6D0(a1, v10, a4, 0);
              break;
            case 26:
              if ( a5 )
                sub_A1E8D0(a1, v10, a4, (*a5)[21]);
              else
                sub_A1E8D0(a1, v10, a4, 0);
              break;
            case 27:
              if ( a5 )
                sub_A1EAC0(a1, v10, a4, (*a5)[22]);
              else
                sub_A1EAC0(a1, v10, a4, 0);
              break;
            case 28:
              if ( a5 )
                sub_A1EC80(a1, v10, a4, (*a5)[23]);
              else
                sub_A1EC80(a1, v10, a4, 0);
              break;
            case 29:
              if ( a5 )
                sub_A1EE50(a1, v10, a4, (*a5)[24]);
              else
                sub_A1EE50(a1, v10, a4, 0);
              break;
            case 30:
              v34 = (*(_BYTE *)(v10 + 1) & 0x7F) == 1;
              if ( a5 )
              {
                v35 = (*a5)[25];
                sub_A188E0(a4, v34);
                v36 = v35;
              }
              else
              {
                sub_A188E0(a4, v34);
                v36 = 0;
              }
              sub_A1BFB0(*a1, 0x2Fu, a4, v36);
              *(_DWORD *)(a4 + 8) = 0;
              break;
            case 31:
              if ( a5 )
                sub_A1F030(a1, v10, a4, (*a5)[26]);
              else
                sub_A1F030(a1, v10, a4, 0);
              break;
            case 32:
              if ( a5 )
                sub_A1F240(a1, v10, a4, (*a5)[27]);
              else
                sub_A1F240(a1, v10, a4, 0);
              break;
            case 33:
              if ( a5 )
                sub_A1F440(a1, v10, a4, (*a5)[28]);
              else
                sub_A1F440(a1, v10, a4, 0);
              break;
            case 34:
              if ( a5 )
                sub_A1F5C0(a1, v10, a4, (*a5)[29]);
              else
                sub_A1F5C0(a1, v10, a4, 0);
              break;
            case 35:
              if ( a5 )
                sub_A1F780(a1, v10, a4, (*a5)[30]);
              else
                sub_A1F780(a1, v10, a4, 0);
              break;
            case 36:
              if ( a5 )
                sub_A1F940(a1, (_BYTE *)v10, a4, (*a5)[31]);
              else
                sub_A1F940(a1, (_BYTE *)v10, a4, 0);
              break;
            default:
              if ( a5 )
                sub_A1C1E0((__int64)a1, v10, a4, **a5);
              else
                sub_A1C1E0((__int64)a1, v10, a4, 0);
              break;
          }
        }
        ++v6;
      }
      while ( v43 != v6 );
    }
  }
}
