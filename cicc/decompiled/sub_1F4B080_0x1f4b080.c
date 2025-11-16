// Function: sub_1F4B080
// Address: 0x1f4b080
//
__int64 __fastcall sub_1F4B080(
        _QWORD *a1,
        _QWORD *a2,
        unsigned int a3,
        _QWORD *a4,
        unsigned int a5,
        unsigned int *a6,
        unsigned int *a7)
{
  __int64 v12; // rdx
  __int64 v13; // r10
  __int64 v14; // rsi
  __int64 v15; // rax
  int v16; // edi
  unsigned int v17; // r8d
  unsigned int v18; // r9d
  unsigned int *v19; // rsi
  _QWORD *v20; // rsi
  __int64 v21; // r15
  __int64 v22; // rax
  unsigned int v23; // r8d
  __int64 v24; // rbx
  __int64 v25; // r11
  unsigned __int16 *v26; // r14
  unsigned int v27; // r13d
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // esi
  __int64 v33; // r10
  unsigned int v34; // eax
  int v35; // eax
  unsigned int v37; // eax
  __int64 v38; // [rsp+0h] [rbp-80h]
  unsigned int v39; // [rsp+Ch] [rbp-74h]
  _QWORD *v40; // [rsp+10h] [rbp-70h]
  _WORD *v41; // [rsp+18h] [rbp-68h]
  unsigned int *v42; // [rsp+20h] [rbp-60h]
  __int64 v43; // [rsp+28h] [rbp-58h]
  __int64 v44; // [rsp+30h] [rbp-50h]
  unsigned int v45; // [rsp+38h] [rbp-48h]
  unsigned int v46; // [rsp+3Ch] [rbp-44h]
  unsigned int v47; // [rsp+3Ch] [rbp-44h]
  __int64 v48; // [rsp+40h] [rbp-40h]
  unsigned int v49; // [rsp+48h] [rbp-38h]
  unsigned int v50; // [rsp+48h] [rbp-38h]
  unsigned int v51; // [rsp+4Ch] [rbp-34h]

  v12 = a1[33];
  v13 = a1[32];
  v40 = a2;
  v14 = a1[35];
  v39 = a5;
  v42 = a6;
  v15 = (v12 - v13) >> 3;
  v16 = v15 * *((_DWORD *)a1 + 72);
  v17 = *(_DWORD *)(v14 + 24LL * (v16 + (unsigned int)*(unsigned __int16 *)(*a2 + 24LL)));
  v18 = *(_DWORD *)(v14 + 24LL * ((unsigned int)*(unsigned __int16 *)(*a4 + 24LL) + v16));
  if ( v18 <= v17 )
  {
    v19 = a7;
    a7 = a6;
    v18 = v17;
    v39 = a3;
    a3 = a5;
    v42 = v19;
    v20 = a4;
    a4 = a2;
    v40 = v20;
  }
  v21 = a4[1];
  v22 = (unsigned int)(v15 + 31) >> 5;
  v48 = a4[2];
  if ( v48 )
  {
    v41 = (_WORD *)a4[2];
    v23 = a3;
    v38 = 4 * v22;
    v48 = 0;
    v51 = v39;
    v45 = 0;
    while ( 1 )
    {
      v24 = v40[1];
      if ( v40[2] )
      {
        v25 = 4LL * (((unsigned int)((v12 - v13) >> 3) + 31) >> 5);
        v26 = (unsigned __int16 *)v40[2];
        v27 = 0;
        while ( 1 )
        {
          v28 = (v12 - v13) >> 3;
          if ( (_DWORD)v28 )
          {
            v29 = 0;
            v30 = 0;
            do
            {
              if ( *(_DWORD *)(v24 + v29) & *(_DWORD *)(v21 + v29) )
              {
                __asm { tzcnt   eax, eax }
                v33 = *(_QWORD *)(v13 + 8LL * (v30 + _EAX));
                if ( v33
                  && *(_DWORD *)(a1[35]
                               + 24LL
                               * (*((_DWORD *)a1 + 72) * (_DWORD)v28
                                + (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v33 + 24LL))) >= v18 )
                {
                  v34 = v23;
                  if ( v27 )
                  {
                    v34 = v27;
                    if ( v23 )
                    {
                      v43 = v33;
                      v44 = v25;
                      v46 = v18;
                      v49 = v23;
                      v34 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a1 + 120LL))(a1, v27, v23);
                      v33 = v43;
                      v25 = v44;
                      v18 = v46;
                      v23 = v49;
                    }
                  }
                  if ( v51 == v34 )
                  {
                    if ( !v48
                      || (v35 = *((_DWORD *)a1 + 72) * ((__int64)(a1[33] - a1[32]) >> 3),
                          *(_DWORD *)(a1[35] + 24LL * (v35 + (unsigned int)*(unsigned __int16 *)(*(_QWORD *)v48 + 24LL))) > *(_DWORD *)(a1[35] + 24LL * ((unsigned int)*(unsigned __int16 *)(*(_QWORD *)v33 + 24LL) + v35))) )
                    {
                      *a7 = v45;
                      *v42 = v27;
                      if ( *(_DWORD *)(a1[35]
                                     + 24LL
                                     * (*(unsigned __int16 *)(*(_QWORD *)v33 + 24LL)
                                      + *((_DWORD *)a1 + 72) * (unsigned int)((__int64)(a1[33] - a1[32]) >> 3))) == v18 )
                        return v33;
                      v48 = v33;
                    }
                  }
                }
                goto LABEL_20;
              }
              v30 += 32;
              v29 += 4;
            }
            while ( (unsigned int)v28 > v30 );
            v27 = *v26;
            v24 += v25;
            ++v26;
            if ( !v27 )
              break;
          }
          else
          {
LABEL_20:
            v27 = *v26;
            v24 += v25;
            ++v26;
            if ( !v27 )
              break;
          }
          v12 = a1[33];
          v13 = a1[32];
        }
      }
      v45 = (unsigned __int16)*v41;
      if ( !*v41 )
        break;
      v51 = (unsigned __int16)*v41;
      if ( v39 )
      {
        v47 = v23;
        v50 = v18;
        v37 = (*(__int64 (__fastcall **)(_QWORD *))(*a1 + 120LL))(a1);
        v23 = v47;
        v18 = v50;
        v51 = v37;
      }
      ++v41;
      v12 = a1[33];
      v13 = a1[32];
      v21 += v38;
    }
  }
  return v48;
}
