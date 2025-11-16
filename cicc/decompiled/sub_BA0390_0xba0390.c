// Function: sub_BA0390
// Address: 0xba0390
//
__int64 __fastcall sub_BA0390(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // dl
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // r13
  int v9; // eax
  int v10; // ecx
  unsigned int i; // r12d
  __int64 *v12; // r14
  __int64 v13; // rdi
  unsigned __int16 v14; // ax
  unsigned int v15; // r12d
  __int64 v16; // rax
  int v17; // eax
  _BYTE *v18; // rax
  _QWORD *v19; // r8
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 result; // rax
  unsigned int v23; // esi
  int v24; // eax
  _QWORD *v25; // rdx
  int v26; // eax
  int v27; // eax
  _BYTE *v28; // rax
  _QWORD *v29; // r8
  _QWORD *v30; // rax
  int v31; // [rsp+8h] [rbp-98h]
  __int64 v32; // [rsp+8h] [rbp-98h]
  __int64 v33; // [rsp+8h] [rbp-98h]
  int v34; // [rsp+10h] [rbp-90h]
  int v35; // [rsp+18h] [rbp-88h]
  _QWORD *v36; // [rsp+18h] [rbp-88h]
  _QWORD *v37; // [rsp+18h] [rbp-88h]
  int v38; // [rsp+24h] [rbp-7Ch]
  __int64 v39[2]; // [rsp+28h] [rbp-78h] BYREF
  _QWORD *v40; // [rsp+38h] [rbp-68h] BYREF
  _QWORD *v41; // [rsp+40h] [rbp-60h] BYREF
  __int64 v42; // [rsp+48h] [rbp-58h]
  __int64 v43; // [rsp+50h] [rbp-50h]
  __int64 v44; // [rsp+58h] [rbp-48h]
  int v45; // [rsp+60h] [rbp-40h]
  int v46; // [rsp+64h] [rbp-3Ch] BYREF
  __int64 v47[7]; // [rsp+68h] [rbp-38h] BYREF

  v39[0] = a1;
  v41 = 0;
  v42 = 0;
  v3 = *(_BYTE *)(a1 - 16);
  if ( (v3 & 2) != 0 )
  {
    v4 = *(_QWORD *)(a1 - 32);
    v5 = v4 + 8LL * *(unsigned int *)(a1 - 24);
  }
  else
  {
    v4 = a1 - 16 - 8LL * ((v3 >> 2) & 0xF);
    v5 = v4 + 8LL * ((*(_WORD *)(a1 - 16) >> 6) & 0xF);
  }
  v43 = v4 + 8;
  v44 = (v5 - (v4 + 8)) >> 3;
  v45 = *(_DWORD *)(a1 + 4);
  v46 = (unsigned __int16)sub_AF2710(a1);
  v6 = sub_AF5140(a1, 0);
  v7 = *(_DWORD *)(a2 + 24);
  v8 = *(_QWORD *)(a2 + 8);
  v47[0] = v6;
  if ( v7 )
  {
    LODWORD(v40) = v45;
    v9 = sub_AFB9D0((int *)&v40, &v46, v47);
    v38 = 1;
    v10 = v7 - 1;
    for ( i = (v7 - 1) & v9; ; i = v10 & v15 )
    {
      v12 = (__int64 *)(v8 + 8LL * i);
      v13 = *v12;
      if ( *v12 == -4096 )
        break;
      if ( v13 != -8192 )
      {
        v31 = v10;
        v35 = v46;
        v14 = sub_AF2710(v13);
        v10 = v31;
        if ( v35 == v14 )
        {
          v34 = v31;
          v16 = sub_AF5140(v13, 0);
          v10 = v31;
          if ( v47[0] == v16 && v45 == *(_DWORD *)(v13 + 4) )
          {
            if ( v42 )
            {
              if ( (*(_BYTE *)(v13 - 16) & 2) != 0 )
                v27 = *(_DWORD *)(v13 - 24);
              else
                v27 = (*(_WORD *)(v13 - 16) >> 6) & 0xF;
              if ( v42 == v27 - 1 )
              {
                v33 = v42;
                v37 = v41;
                v28 = sub_AF15A0((_BYTE *)(v13 - 16));
                v29 = v37;
                v10 = v34;
                v30 = v28 + 8;
                while ( *v29 == *v30 )
                {
                  ++v29;
                  ++v30;
                  if ( &v37[v33] == v29 )
                    goto LABEL_21;
                }
              }
            }
            else
            {
              if ( (*(_BYTE *)(v13 - 16) & 2) != 0 )
                v17 = *(_DWORD *)(v13 - 24);
              else
                v17 = (*(_WORD *)(v13 - 16) >> 6) & 0xF;
              if ( v17 - 1 == v44 )
              {
                v32 = v44;
                v36 = (_QWORD *)v43;
                v18 = sub_AF15A0((_BYTE *)(v13 - 16));
                v19 = v36;
                v20 = v18 + 8;
                v10 = v34;
                v21 = &v36[v32];
                if ( v21 == v36 )
                {
LABEL_21:
                  if ( v12 != (__int64 *)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
                  {
                    result = *v12;
                    if ( *v12 )
                      return result;
                  }
                  break;
                }
                while ( *v19 == *v20 )
                {
                  ++v19;
                  ++v20;
                  if ( v21 == v19 )
                    goto LABEL_21;
                }
              }
            }
          }
        }
        v13 = *v12;
      }
      if ( v13 == -4096 )
        break;
      v15 = v38 + i;
      ++v38;
    }
  }
  if ( !(unsigned __int8)sub_AFC300(a2, v39, &v40) )
  {
    v23 = *(_DWORD *)(a2 + 24);
    v24 = *(_DWORD *)(a2 + 16);
    v25 = v40;
    ++*(_QWORD *)a2;
    v26 = v24 + 1;
    v41 = v25;
    if ( 4 * v26 >= 3 * v23 )
    {
      v23 *= 2;
    }
    else if ( v23 - *(_DWORD *)(a2 + 20) - v26 > v23 >> 3 )
    {
LABEL_28:
      *(_DWORD *)(a2 + 16) = v26;
      if ( *v25 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v25 = v39[0];
      return v39[0];
    }
    sub_B02740(a2, v23);
    sub_AFC300(a2, v39, &v41);
    v25 = v41;
    v26 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_28;
  }
  return v39[0];
}
