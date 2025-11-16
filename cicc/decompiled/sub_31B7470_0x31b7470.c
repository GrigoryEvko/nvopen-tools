// Function: sub_31B7470
// Address: 0x31b7470
//
__int64 __fastcall sub_31B7470(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // r8
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  _BYTE *v11; // rdi
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // r12
  bool v17; // al
  __int64 *v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int v22; // eax
  __int64 v23; // rdx
  unsigned int v24; // ebx
  unsigned __int64 v25; // rsi
  __int64 v26; // rcx
  unsigned int v27; // eax
  unsigned int v28; // r14d
  unsigned int i; // ebx
  unsigned int v30; // eax
  _QWORD *v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // [rsp+0h] [rbp-240h]
  __int64 v40; // [rsp+8h] [rbp-238h]
  __int64 v41; // [rsp+18h] [rbp-228h]
  __int64 v42; // [rsp+20h] [rbp-220h]
  __int64 v43; // [rsp+28h] [rbp-218h]
  unsigned int v44; // [rsp+34h] [rbp-20Ch]
  unsigned int v46; // [rsp+40h] [rbp-200h]
  unsigned int v47; // [rsp+44h] [rbp-1FCh]
  __int64 v48; // [rsp+48h] [rbp-1F8h]
  unsigned __int8 v50; // [rsp+5Bh] [rbp-1E5h]
  unsigned int v51; // [rsp+5Ch] [rbp-1E4h]
  __int64 v52; // [rsp+60h] [rbp-1E0h]
  __int64 v53; // [rsp+68h] [rbp-1D8h]
  _BYTE *v54; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v55; // [rsp+78h] [rbp-1C8h]
  _QWORD *v56; // [rsp+80h] [rbp-1C0h]
  unsigned __int64 v57; // [rsp+88h] [rbp-1B8h]
  _BYTE v58[32]; // [rsp+90h] [rbp-1B0h] BYREF
  __int64 v59; // [rsp+B0h] [rbp-190h]
  unsigned int v60; // [rsp+B8h] [rbp-188h]
  _BYTE *v61; // [rsp+150h] [rbp-F0h] BYREF
  __int64 v62; // [rsp+158h] [rbp-E8h]
  __int64 v63; // [rsp+160h] [rbp-E0h]
  unsigned __int64 v64; // [rsp+168h] [rbp-D8h]

  v42 = *(_QWORD *)sub_3187030(*(_QWORD *)(a2 + 24), *(_QWORD *)(*(_QWORD *)(a2 + 16) + 40LL)) + 312LL;
  v44 = qword_5035968;
  if ( !(_DWORD)qword_5035968 )
  {
    v61 = (_BYTE *)sub_DFB1B0(*(_QWORD *)(a3 + 16));
    v62 = v38;
    v44 = (unsigned int)v61;
  }
  v50 = 0;
  v3 = *(_QWORD *)(a2 + 16);
  v40 = *(_QWORD *)(a2 + 24);
  v39 = v3 + 72;
  v41 = *(_QWORD *)(v3 + 80);
  if ( v41 != v3 + 72 )
  {
    v48 = a1 + 40;
    do
    {
      v4 = v41 - 24;
      if ( !v41 )
        v4 = 0;
      v5 = sub_3186770(v40, v4);
      sub_31C4A00(v58, v5, *(_QWORD *)(a3 + 8));
      v6 = v60;
      v7 = v59;
      if ( v60 )
      {
        v10 = v59 + 24;
        v9 = 0;
        v62 = v59;
        v63 = v59 + 24;
        v61 = v58;
        v64 = 0;
        while ( *(unsigned int *)(v10 + 8) > v9
             && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 8 * v9) + 144LL) == *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v10 + 8 * v9)
                                                                                       + 16LL) )
        {
          v64 = v9 + 1;
          if ( v9 + 1 >= *(unsigned int *)(v10 + 8) )
          {
            v64 = 0;
            v37 = v62 + 112;
            v62 += 88;
            if ( v62 == *((_QWORD *)v61 + 4) + 88LL * *((unsigned int *)v61 + 10) )
              v37 = 0;
            v63 = v37;
          }
          sub_31B7070((__int64 *)&v61);
          v10 = v63;
          v9 = v64;
          if ( !v63 )
            goto LABEL_75;
        }
        v10 = v63;
LABEL_75:
        v11 = v61;
        v7 = v62;
        v8 = v59;
        v6 = v60;
      }
      else
      {
        v8 = v59;
        v9 = 0;
        v10 = 0;
        v11 = v58;
      }
      v54 = v11;
      v55 = v7;
      v56 = (_QWORD *)v10;
      v43 = v8 + 88 * v6;
      v57 = v9;
LABEL_10:
      if ( v43 == v7 )
        goto LABEL_56;
      do
      {
        while ( 1 )
        {
LABEL_11:
          v12 = *(_QWORD *)(*v56 + 8 * v9);
          v13 = *(unsigned int *)(v12 + 16);
          if ( *(_DWORD *)(v12 + 16) )
          {
            v14 = 0;
            while ( v14 != *(_DWORD *)(v12 + 136)
                 && (*(_QWORD *)(*(_QWORD *)(v12 + 72) + 8LL * ((unsigned int)v14 >> 6)) & (1LL << v14)) != 0 )
            {
              if ( v13 == ++v14 )
              {
                v15 = 8 * v13;
                goto LABEL_17;
              }
            }
            v15 = 8 * v14;
          }
          else
          {
            v15 = 0;
          }
LABEL_17:
          v16 = *(_QWORD *)(*(_QWORD *)(v12 + 8) + v15);
          v17 = sub_318B630(v16);
          if ( v16 && v17 && (*(_DWORD *)(v16 + 8) != 37 || sub_318B6C0(v16)) )
          {
            if ( sub_318B670(v16) )
            {
              v16 = sub_318B680(v16);
            }
            else if ( *(_DWORD *)(v16 + 8) == 37 )
            {
              v16 = sub_318B6C0(v16);
            }
          }
          v18 = sub_318EB80(v16);
          v19 = *v18;
          if ( (unsigned int)*(unsigned __int8 *)(*v18 + 8) - 17 <= 1 )
            v19 = *sub_318E560(v18);
          v20 = sub_9208B0(v42, v19);
          v62 = v21;
          v61 = (_BYTE *)v20;
          v46 = sub_CA1930(&v61);
          v23 = v44 % v46;
          v22 = v44 / v46;
          if ( *(_DWORD *)(v12 + 148) / v46 <= v44 / v46 )
            v22 = *(_DWORD *)(v12 + 148) / v46;
          v47 = v22;
          if ( v22 > 1 )
          {
            do
            {
              v24 = *(_DWORD *)(v12 + 16);
              if ( *(_DWORD *)(v12 + 144) == v24 )
                break;
              v25 = v24;
              if ( v24 )
              {
                v26 = 0;
                v27 = *(_DWORD *)(v12 + 136);
                while ( 1 )
                {
                  v28 = v26;
                  if ( *(_DWORD *)(v12 + 136) == v26
                    || (*(_QWORD *)(*(_QWORD *)(v12 + 72) + 8LL * ((unsigned int)v26 >> 6)) & (1LL << v26)) == 0 )
                  {
                    break;
                  }
                  if ( v24 == ++v26 )
                  {
                    v28 = *(_DWORD *)(v12 + 16);
                    break;
                  }
                }
                v23 = v28 + 1;
                if ( v24 > (unsigned int)v23 )
                {
                  v51 = v24 - 2;
                  for ( i = v28; ; ++i )
                  {
                    if ( i >= v27
                      || (v25 = *(_QWORD *)(v12 + 72), v23 = i >> 6, (*(_QWORD *)(v25 + 8 * v23) & (1LL << i)) == 0) )
                    {
                      if ( *(_DWORD *)(v12 + 144) == *(_DWORD *)(v12 + 16) )
                        break;
                      v25 = i;
                      v52 = sub_31C2A20(v12, i, v46 * v47, (unsigned __int8)qword_5035888 ^ 1u);
                      v53 = v23;
                      if ( v23 )
                      {
                        sub_371BA10(&v61, *(_QWORD *)(a2 + 24), *(_QWORD *)(a3 + 16));
                        sub_371BC00(&v61, v52, v53);
                        v25 = (unsigned __int64)&v61;
                        v50 |= sub_318D2B0(v48, (__int64)&v61, a3);
                        sub_371C090(&v61);
                        sub_371BB90(&v61);
                      }
                    }
                    if ( v51 == i )
                      break;
                    v27 = *(_DWORD *)(v12 + 136);
                  }
                }
              }
              v30 = sub_31C5440(v47, v25, v23);
              v23 = v47 >> 1;
              if ( v47 == v30 )
                v30 = v47 >> 1;
              v47 = v30;
            }
            while ( v30 > 1 );
          }
          v9 = v57 + 1;
          v31 = v56;
          v57 = v9;
          v32 = *((unsigned int *)v56 + 2);
          if ( v9 < v32 )
            break;
          v33 = v55;
          v57 = 0;
          v34 = *((unsigned int *)v54 + 10);
          v35 = v55 + 88;
          v55 = v35;
          if ( v35 != *((_QWORD *)v54 + 4) + 88 * v34 )
          {
            v31 = (_QWORD *)(v33 + 112);
            v32 = *(unsigned int *)(v33 + 120);
            v9 = 0;
            v56 = v31;
            break;
          }
          v56 = 0;
          v9 = 0;
          if ( v43 == v35 )
            goto LABEL_57;
        }
        while ( v32 > v9
             && *(_DWORD *)(*(_QWORD *)(*v31 + 8 * v9) + 144LL) == *(_DWORD *)(*(_QWORD *)(*v31 + 8 * v9) + 16LL) )
        {
          sub_31B7350((__int64 *)&v54);
          v31 = v56;
          if ( !v56 )
          {
            v7 = v55;
            v9 = v57;
            goto LABEL_10;
          }
          v9 = v57;
          v32 = *((unsigned int *)v56 + 2);
        }
        if ( v43 != v55 )
          goto LABEL_11;
LABEL_56:
        ;
      }
      while ( v9 );
LABEL_57:
      sub_31C3180(v58);
      v41 = *(_QWORD *)(v41 + 8);
    }
    while ( v39 != v41 );
  }
  return v50;
}
