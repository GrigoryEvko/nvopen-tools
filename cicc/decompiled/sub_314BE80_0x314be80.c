// Function: sub_314BE80
// Address: 0x314be80
//
__int64 __fastcall sub_314BE80(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  bool v5; // dl
  int v6; // ebx
  __int64 v7; // rdx
  int v8; // r12d
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  _QWORD *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  unsigned __int8 v25; // al
  __int64 v26; // rdx
  __int64 v27; // rax
  _QWORD *v28; // rsi
  unsigned __int8 v29; // al
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rsi
  unsigned int *v33; // rdx
  unsigned __int8 v34; // al
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rsi
  unsigned int *v38; // rdx
  unsigned __int8 v39; // al
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rsi
  unsigned int *v43; // rdx
  unsigned __int8 v44; // al
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rsi
  unsigned int *v48; // rdx
  unsigned __int8 v49; // al
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rsi
  unsigned int *v53; // rdx
  unsigned __int8 v54; // al
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // rcx
  _QWORD *v58; // rdx
  __int64 v59; // r8
  __int64 v60; // rsi
  int v61; // edx
  __int64 v62; // rdi
  unsigned __int8 v63; // al
  __int64 v64; // rdx
  _QWORD *v65; // rcx
  _QWORD *v66; // rdx
  int v67; // edx
  int v68; // edi
  unsigned int v69; // [rsp+Ch] [rbp-64h]
  __int64 v70; // [rsp+10h] [rbp-60h]
  __int64 v71; // [rsp+18h] [rbp-58h]
  __int64 v72; // [rsp+20h] [rbp-50h]
  __int64 v73; // [rsp+28h] [rbp-48h]
  __int64 v74; // [rsp+30h] [rbp-40h]
  char v75; // [rsp+3Ah] [rbp-36h]
  char v76; // [rsp+3Bh] [rbp-35h]
  int v77; // [rsp+3Ch] [rbp-34h]

  result = *(unsigned __int8 *)(a1 - 16);
  v5 = (*(_BYTE *)(a1 - 16) & 2) != 0;
  if ( (*(_BYTE *)(a1 - 16) & 2) != 0 )
    v77 = *(_DWORD *)(a1 - 24);
  else
    v77 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
  if ( v77 <= 0 )
    return result;
  v72 = 0;
  v6 = 0;
  v74 = a1 - 16;
  v71 = 0;
  v73 = 0;
  v70 = 0;
  v69 = 0;
  v75 = 0;
  v76 = 0;
  while ( 1 )
  {
    if ( v5 )
      v7 = *(_QWORD *)(a1 - 32);
    else
      v7 = v74 - 8LL * (((unsigned __int8)result >> 2) & 0xF);
    v8 = v6 + 1;
    v9 = *(_QWORD *)(v7 + 8LL * v6);
    v10 = sub_B91420(v9);
    if ( v11 == 10 && *(_QWORD *)v10 == 0x7261507473726966LL && *(_WORD *)(v10 + 8) == 28001 )
    {
      v34 = *(_BYTE *)(a1 - 16);
      if ( (v34 & 2) != 0 )
        v35 = *(_QWORD *)(a1 - 32);
      else
        v35 = v74 - 8LL * ((v34 >> 2) & 0xF);
      v36 = *(_QWORD *)(*(_QWORD *)(v35 + 8LL * v8) + 136LL);
      v37 = *(unsigned int *)(v36 + 32);
      v38 = *(unsigned int **)(v36 + 24);
      if ( (unsigned int)v37 > 0x40 )
      {
        v37 = *v38;
      }
      else if ( (_DWORD)v37 )
      {
        v37 = (__int64)((_QWORD)v38 << (64 - (unsigned __int8)v37)) >> (64 - (unsigned __int8)v37);
      }
      result = sub_39367E0(a2, v37, v38);
      goto LABEL_43;
    }
    v12 = sub_B91420(v9);
    if ( v13 == 9 && *(_QWORD *)v12 == 0x6D617261506D756ELL && *(_BYTE *)(v12 + 8) == 115 )
      break;
    v14 = sub_B91420(v9);
    if ( v15 == 12 && *(_QWORD *)v14 == 0x7465527473726966LL && *(_DWORD *)(v14 + 8) == 1433301621 )
    {
      v39 = *(_BYTE *)(a1 - 16);
      if ( (v39 & 2) != 0 )
        v40 = *(_QWORD *)(a1 - 32);
      else
        v40 = v74 - 8LL * ((v39 >> 2) & 0xF);
      v41 = *(_QWORD *)(*(_QWORD *)(v40 + 8LL * v8) + 136LL);
      v42 = *(unsigned int *)(v41 + 32);
      v43 = *(unsigned int **)(v41 + 24);
      if ( (unsigned int)v42 > 0x40 )
      {
        v42 = *v43;
      }
      else if ( (_DWORD)v42 )
      {
        v42 = (__int64)((_QWORD)v43 << (64 - (unsigned __int8)v42)) >> (64 - (unsigned __int8)v42);
      }
      result = sub_3936800(a2, v42, v43);
    }
    else
    {
      v16 = sub_B91420(v9);
      if ( v17 == 11
        && *(_QWORD *)v16 == 0x7465527473726966LL
        && *(_WORD *)(v16 + 8) == 29301
        && *(_BYTE *)(v16 + 10) == 110 )
      {
        v44 = *(_BYTE *)(a1 - 16);
        if ( (v44 & 2) != 0 )
          v45 = *(_QWORD *)(a1 - 32);
        else
          v45 = v74 - 8LL * ((v44 >> 2) & 0xF);
        v46 = *(_QWORD *)(*(_QWORD *)(v45 + 8LL * v8) + 136LL);
        v47 = *(unsigned int *)(v46 + 32);
        v48 = *(unsigned int **)(v46 + 24);
        if ( (unsigned int)v47 > 0x40 )
        {
          v47 = *v48;
        }
        else if ( (_DWORD)v47 )
        {
          v47 = (__int64)((_QWORD)v48 << (64 - (unsigned __int8)v47)) >> (64 - (unsigned __int8)v47);
        }
        result = sub_39367F0(a2, v47, v48);
      }
      else
      {
        v18 = sub_B91420(v9);
        if ( v19 == 11
          && *(_QWORD *)v18 == 0x78614D6C61636F6CLL
          && *(_WORD *)(v18 + 8) == 25938
          && *(_BYTE *)(v18 + 10) == 103 )
        {
          v49 = *(_BYTE *)(a1 - 16);
          if ( (v49 & 2) != 0 )
            v50 = *(_QWORD *)(a1 - 32);
          else
            v50 = v74 - 8LL * ((v49 >> 2) & 0xF);
          v51 = *(_QWORD *)(*(_QWORD *)(v50 + 8LL * v8) + 136LL);
          v52 = *(unsigned int *)(v51 + 32);
          v53 = *(unsigned int **)(v51 + 24);
          if ( (unsigned int)v52 > 0x40 )
          {
            v52 = *v53;
          }
          else if ( (_DWORD)v52 )
          {
            v52 = (__int64)((_QWORD)v53 << (64 - (unsigned __int8)v52)) >> (64 - (unsigned __int8)v52);
          }
          result = sub_3936840(a2, v52, v53);
        }
        else
        {
          v20 = (_QWORD *)sub_B91420(v9);
          if ( v21 == 8 && *v20 == 0x5268637461726373LL )
          {
            v54 = *(_BYTE *)(a1 - 16);
            if ( (v54 & 2) != 0 )
              v55 = *(_QWORD *)(a1 - 32);
            else
              v55 = v74 - 8LL * ((v54 >> 2) & 0xF);
            result = *(_QWORD *)(v55 + 8LL * v8);
            if ( result && (v56 = *(_QWORD *)(result + 136)) != 0 )
            {
              result = *(_QWORD *)(v56 + 24);
              if ( *(_DWORD *)(v56 + 32) > 0x40u )
                result = *(_QWORD *)result;
              v8 = v6 + 2;
              v57 = *(_QWORD *)(*(_QWORD *)(v55 + 8LL * (v6 + 2)) + 136LL);
              v58 = *(_QWORD **)(v57 + 24);
              if ( *(_DWORD *)(v57 + 32) > 0x40u )
                v58 = (_QWORD *)*v58;
              if ( (int)result > (int)v58 )
              {
                v76 = 1;
              }
              else
              {
                v59 = v73;
                v60 = v72;
                v61 = (_DWORD)v58 + 1;
                v62 = v71;
                do
                {
                  if ( (int)result <= 63 )
                  {
                    v60 |= 1LL << result;
                  }
                  else if ( (int)result > 127 )
                  {
                    if ( (int)result > 191 )
                    {
                      if ( (int)result <= 255 )
                        v70 |= 1LL << ((unsigned __int8)result + 64);
                    }
                    else
                    {
                      v59 |= 1LL << ((unsigned __int8)result + 0x80);
                    }
                  }
                  else
                  {
                    v62 |= 1LL << ((unsigned __int8)result - 64);
                  }
                  result = (unsigned int)(result + 1);
                }
                while ( (_DWORD)result != v61 );
                v73 = v59;
                v72 = v60;
                v71 = v62;
                v76 = 1;
              }
            }
            else
            {
              v72 = 0;
              v71 = 0;
              v73 = 0;
              v70 = 0;
              v76 = 1;
            }
          }
          else
          {
            v22 = sub_B91420(v9);
            if ( v23 == 9 && *(_QWORD *)v22 == 0x4368637461726373LL && *(_BYTE *)(v22 + 8) == 66 )
            {
              v63 = *(_BYTE *)(a1 - 16);
              if ( (v63 & 2) != 0 )
                v64 = *(_QWORD *)(a1 - 32);
              else
                v64 = v74 - 8LL * ((v63 >> 2) & 0xF);
              result = *(_QWORD *)(v64 + 8LL * v8);
              if ( result && (result = *(_QWORD *)(result + 136)) != 0 )
              {
                v65 = *(_QWORD **)(result + 24);
                if ( *(_DWORD *)(result + 32) > 0x40u )
                  v65 = (_QWORD *)*v65;
                v8 = v6 + 2;
                result = *(_QWORD *)(*(_QWORD *)(v64 + 8LL * (v6 + 2)) + 136LL);
                v66 = *(_QWORD **)(result + 24);
                if ( *(_DWORD *)(result + 32) > 0x40u )
                  v66 = (_QWORD *)*v66;
                if ( (int)v65 <= (int)v66 )
                {
                  LODWORD(result) = v69;
                  v67 = (_DWORD)v66 + 1;
                  do
                  {
                    v68 = 1 << (char)v65;
                    LODWORD(v65) = (_DWORD)v65 + 1;
                    result = v68 | (unsigned int)result;
                  }
                  while ( (_DWORD)v65 != v67 );
                  v69 = result;
                }
                v75 = 1;
              }
              else
              {
                v69 = 0;
                v75 = 1;
              }
            }
            else
            {
              result = sub_B91420(v9);
              if ( v24 == 10 && *(_QWORD *)result == 0x69747265706F7270LL && *(_WORD *)(result + 8) == 29541 )
              {
                v25 = *(_BYTE *)(a1 - 16);
                if ( (v25 & 2) != 0 )
                  v26 = *(_QWORD *)(a1 - 32);
                else
                  v26 = v74 - 8LL * ((v25 >> 2) & 0xF);
                v27 = *(_QWORD *)(*(_QWORD *)(v26 + 8LL * v8) + 136LL);
                v28 = *(_QWORD **)(v27 + 24);
                if ( *(_DWORD *)(v27 + 32) > 0x40u )
                  v28 = (_QWORD *)*v28;
                result = sub_3936850(a2, v28);
              }
            }
          }
        }
      }
    }
LABEL_43:
    v6 = v8 + 1;
    if ( v77 <= v8 + 1 )
      goto LABEL_32;
LABEL_44:
    LOBYTE(result) = *(_BYTE *)(a1 - 16);
    v5 = (result & 2) != 0;
  }
  v29 = *(_BYTE *)(a1 - 16);
  if ( (v29 & 2) != 0 )
    v30 = *(_QWORD *)(a1 - 32);
  else
    v30 = v74 - 8LL * ((v29 >> 2) & 0xF);
  v31 = *(_QWORD *)(*(_QWORD *)(v30 + 8LL * v8) + 136LL);
  v32 = *(unsigned int *)(v31 + 32);
  v33 = *(unsigned int **)(v31 + 24);
  if ( (unsigned int)v32 > 0x40 )
  {
    v32 = *v33;
  }
  else if ( (_DWORD)v32 )
  {
    v32 = (__int64)((_QWORD)v33 << (64 - (unsigned __int8)v32)) >> (64 - (unsigned __int8)v32);
  }
  v6 += 2;
  result = sub_39367D0(a2, v32, v33);
  if ( v77 > v6 )
    goto LABEL_44;
LABEL_32:
  if ( v75 )
    result = sub_3936810(a2, v69);
  if ( v76 )
    return sub_3936820(a2, v70, v73, v71, v72);
  return result;
}
