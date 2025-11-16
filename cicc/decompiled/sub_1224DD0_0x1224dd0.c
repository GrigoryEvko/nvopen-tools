// Function: sub_1224DD0
// Address: 0x1224dd0
//
__int64 __fastcall sub_1224DD0(_QWORD **a1, __int64 *a2, __int64 *a3)
{
  unsigned __int64 v3; // r14
  char v4; // r12
  __int64 result; // rax
  __int64 *v6; // r15
  bool v7; // zf
  __int64 v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r11
  unsigned __int64 v14; // rdx
  __int64 *v15; // rax
  int v16; // r14d
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r15
  int v20; // eax
  unsigned int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // r13
  int v27; // eax
  int v28; // eax
  __int64 v29; // [rsp+8h] [rbp-1B8h]
  bool v30; // [rsp+1Fh] [rbp-1A1h]
  unsigned int v33; // [rsp+28h] [rbp-198h]
  __int64 *v34; // [rsp+38h] [rbp-188h] BYREF
  __int64 v35; // [rsp+40h] [rbp-180h] BYREF
  __int64 v36; // [rsp+48h] [rbp-178h] BYREF
  const char *v37; // [rsp+50h] [rbp-170h] BYREF
  __int16 v38; // [rsp+70h] [rbp-150h]
  const char *v39; // [rsp+80h] [rbp-140h] BYREF
  __int64 v40; // [rsp+88h] [rbp-138h]
  _BYTE v41[16]; // [rsp+90h] [rbp-130h] BYREF
  char v42; // [rsp+A0h] [rbp-120h]
  char v43; // [rsp+A1h] [rbp-11Fh]

  v3 = (unsigned __int64)a1[29];
  v34 = 0;
  v43 = 1;
  v39 = "expected type";
  v42 = 3;
  v4 = sub_12190A0((__int64)a1, &v34, (int *)&v39, 0);
  result = 1;
  if ( !v4 )
  {
    v6 = v34;
    v30 = *((_BYTE *)v34 + 8) != 13 && *((_BYTE *)v34 + 8) != 7;
    if ( v30 )
    {
      v7 = *((_DWORD *)a1 + 60) == 6;
      v39 = v41;
      v40 = 0x1000000000LL;
      if ( v7 )
      {
        while ( 1 )
        {
          v8 = 6;
          if ( (unsigned __int8)sub_120AFE0((__int64)a1, 6, "expected '[' in phi value list") )
            break;
          v8 = (__int64)v34;
          if ( (unsigned __int8)sub_1224B80(a1, (__int64)v34, &v35, a3) )
            break;
          v8 = 4;
          if ( (unsigned __int8)sub_120AFE0((__int64)a1, 4, "expected ',' after insertelement value") )
            break;
          v8 = sub_BCB130(*a1);
          if ( (unsigned __int8)sub_1224B80(a1, v8, &v36, a3) )
            break;
          v8 = 7;
          v4 = sub_120AFE0((__int64)a1, 7, "expected ']' in phi value list");
          if ( v4 )
            break;
          v11 = (unsigned int)v40;
          v12 = v36;
          v13 = v35;
          v14 = (unsigned int)v40 + 1LL;
          if ( v14 > HIDWORD(v40) )
          {
            v29 = v35;
            sub_C8D5F0((__int64)&v39, v41, v14, 0x10u, v9, v10);
            v11 = (unsigned int)v40;
            v13 = v29;
          }
          v15 = (__int64 *)&v39[16 * v11];
          v15[1] = v12;
          *v15 = v13;
          v7 = *((_DWORD *)a1 + 60) == 4;
          v16 = v40 + 1;
          LODWORD(v40) = v40 + 1;
          if ( !v7 )
          {
            v6 = v34;
            goto LABEL_15;
          }
          v28 = sub_1205200((__int64)(a1 + 22));
          *((_DWORD *)a1 + 60) = v28;
          if ( v28 == 511 )
          {
            v6 = v34;
            v16 = v40;
            v4 = v30;
            goto LABEL_15;
          }
        }
        result = 1;
      }
      else
      {
        v16 = 0;
LABEL_15:
        v38 = 257;
        v17 = sub_BD2DA0(80);
        v18 = v17;
        if ( v17 )
        {
          sub_B44260(v17, (__int64)v6, 55, 0x8000000u, 0, 0);
          *(_DWORD *)(v18 + 72) = v16;
          sub_BD6B50((unsigned __int8 *)v18, &v37);
          sub_BD2A10(v18, *(_DWORD *)(v18 + 72), 1);
        }
        v19 = 0;
        v8 = 16LL * (unsigned int)v40;
        if ( (_DWORD)v40 )
        {
          do
          {
            v25 = *(_QWORD *)&v39[v19 + 8];
            v26 = *(_QWORD *)&v39[v19];
            v27 = *(_DWORD *)(v18 + 4) & 0x7FFFFFF;
            if ( v27 == *(_DWORD *)(v18 + 72) )
            {
              sub_B48D90(v18);
              v27 = *(_DWORD *)(v18 + 4) & 0x7FFFFFF;
            }
            v20 = (v27 + 1) & 0x7FFFFFF;
            v21 = v20 | *(_DWORD *)(v18 + 4) & 0xF8000000;
            v22 = *(_QWORD *)(v18 - 8) + 32LL * (unsigned int)(v20 - 1);
            *(_DWORD *)(v18 + 4) = v21;
            if ( *(_QWORD *)v22 )
            {
              v23 = *(_QWORD *)(v22 + 8);
              **(_QWORD **)(v22 + 16) = v23;
              if ( v23 )
                *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
            }
            *(_QWORD *)v22 = v26;
            if ( v26 )
            {
              v24 = *(_QWORD *)(v26 + 16);
              *(_QWORD *)(v22 + 8) = v24;
              if ( v24 )
                *(_QWORD *)(v24 + 16) = v22 + 8;
              *(_QWORD *)(v22 + 16) = v26 + 16;
              *(_QWORD *)(v26 + 16) = v22;
            }
            v19 += 16;
            *(_QWORD *)(*(_QWORD *)(v18 - 8)
                      + 32LL * *(unsigned int *)(v18 + 72)
                      + 8LL * ((*(_DWORD *)(v18 + 4) & 0x7FFFFFFu) - 1)) = v25;
          }
          while ( v8 != v19 );
        }
        *a2 = v18;
        result = 2 * (unsigned int)(v4 != 0);
      }
      if ( v39 != v41 )
      {
        v33 = result;
        _libc_free(v39, v8);
        return v33;
      }
    }
    else
    {
      v43 = 1;
      v39 = "phi node must have first class type";
      v42 = 3;
      sub_11FD800((__int64)(a1 + 22), v3, (__int64)&v39, 1);
      return 1;
    }
  }
  return result;
}
