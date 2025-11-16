// Function: sub_2B19930
// Address: 0x2b19930
//
__int64 __fastcall sub_2B19930(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _DWORD *a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v8; // r10
  __int64 v9; // r13
  __int64 v10; // r12
  _DWORD *v11; // rbx
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r9
  __int64 v15; // r10
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // r10
  __int64 v20; // rcx
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v27; // rsi
  _DWORD *v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rdx
  _DWORD *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rsi
  __int64 v34; // r11
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v39; // rcx
  _DWORD *v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // rcx
  _DWORD *v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rsi
  __int64 v53; // r10
  __int64 v54; // r11
  __int64 v55; // rdi
  __int64 v56; // rdx
  __int64 v57; // rsi
  _DWORD *v58; // r8
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rsi
  __int64 v62; // rax
  __int64 v63; // r8
  _DWORD *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rdi
  __int64 v67; // r8
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // rdx
  int v72; // [rsp+8h] [rbp-58h]
  __int64 v74; // [rsp+18h] [rbp-48h]
  __int64 v75; // [rsp+18h] [rbp-48h]
  int v76; // [rsp+18h] [rbp-48h]
  __int64 v78; // [rsp+28h] [rbp-38h]

  result = a5;
  v8 = a2;
  v9 = a3;
  v10 = a1;
  v11 = a6;
  if ( a7 <= a5 )
    result = a7;
  if ( result >= a4 )
  {
LABEL_32:
    v39 = (v8 - v10) >> 4;
    if ( v8 - v10 > 0 )
    {
      v40 = v11;
      v41 = v10;
      do
      {
        v42 = *(_QWORD *)(v41 + 8);
        v40 += 4;
        v41 += 16;
        *((_QWORD *)v40 - 1) = v42;
        *(v40 - 3) = *(_DWORD *)(v41 - 12);
        *(v40 - 4) = *(_DWORD *)(v41 - 16);
        --v39;
      }
      while ( v39 );
      v43 = 16;
      if ( v8 - v10 > 0 )
        v43 = v8 - v10;
      result = (__int64)v11 + v43;
      if ( v11 != (_DWORD *)result )
      {
        while ( v9 != v8 )
        {
          if ( *(_DWORD *)(v8 + 4) < v11[1] )
          {
            v44 = *(_QWORD *)(v8 + 8);
            v10 += 16;
            v8 += 16;
            *(_QWORD *)(v10 - 8) = v44;
            *(_DWORD *)(v10 - 12) = *(_DWORD *)(v8 - 12);
            *(_DWORD *)(v10 - 16) = *(_DWORD *)(v8 - 16);
            if ( (_DWORD *)result == v11 )
              break;
          }
          else
          {
            v45 = *((_QWORD *)v11 + 1);
            v11 += 4;
            v10 += 16;
            *(_QWORD *)(v10 - 8) = v45;
            *(_DWORD *)(v10 - 12) = *(v11 - 3);
            *(_DWORD *)(v10 - 16) = *(v11 - 4);
            if ( (_DWORD *)result == v11 )
              break;
          }
        }
        if ( v11 != (_DWORD *)result )
        {
          result -= (__int64)v11;
          v46 = result >> 4;
          if ( result > 0 )
          {
            do
            {
              v47 = *((_QWORD *)v11 + 1);
              v10 += 16;
              v11 += 4;
              *(_QWORD *)(v10 - 8) = v47;
              *(_DWORD *)(v10 - 12) = *(v11 - 3);
              result = (unsigned int)*(v11 - 4);
              *(_DWORD *)(v10 - 16) = result;
              --v46;
            }
            while ( v46 );
          }
        }
      }
    }
  }
  else
  {
    if ( a7 < a5 )
    {
      v12 = a4;
      v13 = a5;
      v14 = a2;
      v15 = a1;
      if ( a5 >= a4 )
        goto LABEL_14;
LABEL_6:
      v16 = v15 + 16 * (v12 / 2);
      v17 = sub_2B0EF00(v14, a3, v16);
      v20 = v12 / 2;
      v78 = v17;
      v12 -= v12 / 2;
      v21 = (v17 - v18) >> 4;
      if ( v12 <= v21 )
        goto LABEL_15;
LABEL_7:
      if ( a7 >= v21 )
      {
        v22 = v16;
        if ( !v21 )
          goto LABEL_9;
        v54 = v18 - v16;
        v55 = v78 - v18;
        v56 = (v18 - v16) >> 4;
        v57 = (v78 - v18) >> 4;
        if ( v78 - v18 <= 0 )
        {
          if ( v54 <= 0 )
            goto LABEL_9;
          v61 = 0;
          v55 = 0;
        }
        else
        {
          v58 = a6;
          v59 = v18;
          do
          {
            v60 = *(_QWORD *)(v59 + 8);
            v58 += 4;
            v59 += 16;
            *((_QWORD *)v58 - 1) = v60;
            *(v58 - 3) = *(_DWORD *)(v59 - 12);
            *(v58 - 4) = *(_DWORD *)(v59 - 16);
            --v57;
          }
          while ( v57 );
          v56 = (v18 - v16) >> 4;
          if ( v55 <= 0 )
            v55 = 16;
          v61 = v55 >> 4;
          if ( v54 <= 0 )
            goto LABEL_69;
        }
        v62 = v78;
        do
        {
          v63 = *(_QWORD *)(v18 - 8);
          v18 -= 16;
          v62 -= 16;
          *(_QWORD *)(v62 + 8) = v63;
          *(_DWORD *)(v62 + 4) = *(_DWORD *)(v18 + 4);
          *(_DWORD *)v62 = *(_DWORD *)v18;
          --v56;
        }
        while ( v56 );
LABEL_69:
        if ( v55 <= 0 )
        {
          v22 = v16;
        }
        else
        {
          v64 = a6;
          v65 = v16;
          v66 = v61;
          do
          {
            v67 = *((_QWORD *)v64 + 1);
            v65 += 16;
            v64 += 4;
            *(_QWORD *)(v65 - 8) = v67;
            *(_DWORD *)(v65 - 12) = *(v64 - 3);
            *(_DWORD *)(v65 - 16) = *(v64 - 4);
            --v66;
          }
          while ( v66 );
          v68 = 16 * v61;
          if ( v61 <= 0 )
            v68 = 16;
          v22 = v16 + v68;
        }
        goto LABEL_9;
      }
      while ( 1 )
      {
LABEL_15:
        if ( a7 < v12 )
        {
          v72 = v19;
          v76 = v20;
          v22 = sub_2B0A8B0(v16, v18, v78);
          LODWORD(v20) = v76;
          LODWORD(v19) = v72;
          goto LABEL_9;
        }
        v22 = v78;
        if ( !v12 )
          goto LABEL_9;
        v24 = v78 - v18;
        v25 = v18 - v16;
        v26 = (v78 - v18) >> 4;
        v27 = (v18 - v16) >> 4;
        if ( v18 - v16 <= 0 )
        {
          if ( v24 <= 0 )
            goto LABEL_9;
          v31 = a6;
          v32 = 0;
          v25 = 0;
        }
        else
        {
          v75 = v24 >> 4;
          v28 = a6;
          v29 = v16;
          do
          {
            v30 = *(_QWORD *)(v29 + 8);
            v28 += 4;
            v29 += 16;
            *((_QWORD *)v28 - 1) = v30;
            *(v28 - 3) = *(_DWORD *)(v29 - 12);
            *(v28 - 4) = *(_DWORD *)(v29 - 16);
            --v27;
          }
          while ( v27 );
          v26 = v75;
          if ( v25 <= 0 )
            v25 = 16;
          v31 = (_DWORD *)((char *)a6 + v25);
          v32 = v25 >> 4;
          if ( v78 - v18 <= 0 )
            goto LABEL_25;
        }
        v33 = v16;
        do
        {
          v34 = *(_QWORD *)(v18 + 8);
          v33 += 16;
          v18 += 16;
          *(_QWORD *)(v33 - 8) = v34;
          *(_DWORD *)(v33 - 12) = *(_DWORD *)(v18 - 12);
          *(_DWORD *)(v33 - 16) = *(_DWORD *)(v18 - 16);
          --v26;
        }
        while ( v26 );
LABEL_25:
        if ( v25 <= 0 )
        {
          v22 = v78;
        }
        else
        {
          v35 = v78;
          v36 = v32;
          do
          {
            v37 = *((_QWORD *)v31 - 1);
            v31 -= 4;
            v35 -= 16;
            *(_QWORD *)(v35 + 8) = v37;
            *(_DWORD *)(v35 + 4) = v31[1];
            *(_DWORD *)v35 = *v31;
            --v36;
          }
          while ( v36 );
          v38 = -16 * v32;
          if ( v32 <= 0 )
            v38 = -16;
          v22 = v78 + v38;
        }
LABEL_9:
        v13 -= v21;
        v74 = v22;
        sub_2B19930(v19, v16, v22, v20, v21, (_DWORD)a6, a7);
        v23 = v13;
        result = v74;
        if ( a7 <= v13 )
          v23 = a7;
        if ( v23 >= v12 )
        {
          v9 = a3;
          v11 = a6;
          v10 = v74;
          v8 = v78;
          goto LABEL_32;
        }
        if ( a7 >= v13 )
        {
          v9 = a3;
          v11 = a6;
          v10 = v74;
          v8 = v78;
          break;
        }
        v14 = v78;
        v15 = v74;
        if ( v13 < v12 )
          goto LABEL_6;
LABEL_14:
        v21 = v13 / 2;
        v78 = v14 + 16 * (v13 / 2);
        v16 = sub_2B0EF50(v15, v14, v78);
        v20 = (v16 - v19) >> 4;
        v12 -= v20;
        if ( v12 > v13 / 2 )
          goto LABEL_7;
      }
    }
    v48 = v9 - v8;
    v49 = (v9 - v8) >> 4;
    if ( v9 - v8 > 0 )
    {
      v50 = v11;
      v51 = v8;
      do
      {
        v52 = *(_QWORD *)(v51 + 8);
        v50 += 4;
        v51 += 16;
        *((_QWORD *)v50 - 1) = v52;
        *(v50 - 3) = *(_DWORD *)(v51 - 12);
        *(v50 - 4) = *(_DWORD *)(v51 - 16);
        --v49;
      }
      while ( v49 );
      if ( v48 <= 0 )
        v48 = 16;
      result = (__int64)v11 + v48;
      if ( v8 == v10 )
      {
        v71 = v48 >> 4;
        while ( 1 )
        {
          result -= 16;
          *(_QWORD *)(v9 - 8) = v52;
          v9 -= 16;
          *(_DWORD *)(v9 + 4) = *(_DWORD *)(result + 4);
          *(_DWORD *)v9 = *(_DWORD *)result;
          if ( !--v71 )
            break;
          v52 = *(_QWORD *)(result - 8);
        }
      }
      else if ( v11 != (_DWORD *)result )
      {
        v53 = v8 - 16;
        while ( 1 )
        {
          result -= 16;
          v9 -= 16;
          if ( *(_DWORD *)(result + 4) < *(_DWORD *)(v53 + 4) )
            break;
LABEL_59:
          *(_QWORD *)(v9 + 8) = *(_QWORD *)(result + 8);
          *(_DWORD *)(v9 + 4) = *(_DWORD *)(result + 4);
          *(_DWORD *)v9 = *(_DWORD *)result;
          if ( v11 == (_DWORD *)result )
            return result;
        }
        while ( 1 )
        {
          *(_QWORD *)(v9 + 8) = *(_QWORD *)(v53 + 8);
          *(_DWORD *)(v9 + 4) = *(_DWORD *)(v53 + 4);
          *(_DWORD *)v9 = *(_DWORD *)v53;
          if ( v53 == v10 )
            break;
          v53 -= 16;
          v9 -= 16;
          if ( *(_DWORD *)(result + 4) >= *(_DWORD *)(v53 + 4) )
            goto LABEL_59;
        }
        result += 16;
        v69 = (result - (__int64)v11) >> 4;
        if ( result - (__int64)v11 > 0 )
        {
          do
          {
            v70 = *(_QWORD *)(result - 8);
            result -= 16;
            v9 -= 16;
            *(_QWORD *)(v9 + 8) = v70;
            *(_DWORD *)(v9 + 4) = *(_DWORD *)(result + 4);
            *(_DWORD *)v9 = *(_DWORD *)result;
            --v69;
          }
          while ( v69 );
        }
      }
    }
  }
  return result;
}
