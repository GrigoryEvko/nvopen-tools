// Function: sub_22CAB90
// Address: 0x22cab90
//
__int64 __fastcall sub_22CAB90(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  _BYTE *v7; // r14
  unsigned __int8 *v8; // r15
  unsigned int v12; // eax
  unsigned __int8 v13; // dl
  unsigned int v14; // esi
  unsigned __int8 v15; // r15
  unsigned int v16; // eax
  unsigned int v17; // esi
  _BYTE *v18; // rsi
  __int64 *v19; // rdi
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 *v23; // rax
  unsigned __int64 v24; // rax
  __int64 *v25; // rax
  unsigned int v26; // [rsp+0h] [rbp-1A0h]
  _BYTE *v27; // [rsp+18h] [rbp-188h]
  unsigned __int8 v29; // [rsp+30h] [rbp-170h]
  unsigned int v30; // [rsp+30h] [rbp-170h]
  __int64 v32[2]; // [rsp+40h] [rbp-160h] BYREF
  __int64 v33; // [rsp+50h] [rbp-150h] BYREF
  __int64 v34[2]; // [rsp+60h] [rbp-140h] BYREF
  __int64 v35; // [rsp+70h] [rbp-130h] BYREF
  __int64 v36[2]; // [rsp+80h] [rbp-120h] BYREF
  __int64 v37; // [rsp+90h] [rbp-110h] BYREF
  __int64 v38[2]; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v39; // [rsp+B0h] [rbp-F0h] BYREF
  __int64 v40[2]; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v41; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v42[2]; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v43; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v44[2]; // [rsp+100h] [rbp-A0h] BYREF
  __int64 v45[2]; // [rsp+110h] [rbp-90h] BYREF
  __int64 v46; // [rsp+120h] [rbp-80h] BYREF
  unsigned int v47; // [rsp+128h] [rbp-78h]
  __int64 v48; // [rsp+130h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+138h] [rbp-68h]
  unsigned __int8 v50[8]; // [rsp+140h] [rbp-60h] BYREF
  __int64 v51; // [rsp+148h] [rbp-58h] BYREF
  unsigned int v52; // [rsp+150h] [rbp-50h]
  __int64 v53; // [rsp+158h] [rbp-48h] BYREF
  unsigned int v54; // [rsp+160h] [rbp-40h]
  char v55; // [rsp+168h] [rbp-38h]

  v7 = *(_BYTE **)(a5 - 64);
  if ( *v7 <= 0x15u )
  {
    v27 = *(_BYTE **)(a5 - 32);
    if ( *v27 <= 0x15u )
    {
      v8 = *(unsigned __int8 **)(a5 - 96);
      if ( sub_98EF80(v8, *(_QWORD *)(*a2 + 240), 0, 0, 0) )
      {
        sub_22C9ED0((__int64)v50, *a2, a3, (__int64)v8, 1u, 0, 0);
        v29 = v50[0];
        if ( v50[0] == 4
          || (v12 = sub_BCB060(*(_QWORD *)(a3 + 8)), v13 = v29, v14 = v12, v29 == 5)
          && (v30 = v12, v23 = sub_9876C0(&v51), v13 = v50[0], v14 = v30, v23) )
        {
          v47 = v52;
          if ( v52 > 0x40 )
            sub_C43780((__int64)&v46, (const void **)&v51);
          else
            v46 = v51;
          v49 = v54;
          if ( v54 > 0x40 )
            sub_C43780((__int64)&v48, (const void **)&v53);
          else
            v48 = v53;
        }
        else if ( v13 == 2 )
        {
          sub_AD8380((__int64)&v46, v51);
        }
        else if ( v13 )
        {
          sub_AADB10((__int64)&v46, v14, 1);
        }
        else
        {
          sub_AADB10((__int64)&v46, v14, 0);
        }
        sub_AB2160((__int64)v32, a4, (__int64)&v46, 0);
        sub_969240(&v48);
        sub_969240(&v46);
        if ( v55 )
        {
          v55 = 0;
          sub_22C0090(v50);
        }
        sub_22C9ED0((__int64)v50, *a2, a3, (__int64)v8, 0, 0, 0);
        v15 = v50[0];
        if ( v50[0] == 4
          || (v16 = sub_BCB060(*(_QWORD *)(a3 + 8)), v17 = v16, v15 == 5)
          && (v26 = v16, v25 = sub_9876C0(&v51), v15 = v50[0], v17 = v26, v25) )
        {
          v47 = v52;
          if ( v52 > 0x40 )
            sub_C43780((__int64)&v46, (const void **)&v51);
          else
            v46 = v51;
          v49 = v54;
          if ( v54 > 0x40 )
            sub_C43780((__int64)&v48, (const void **)&v53);
          else
            v48 = v53;
        }
        else if ( v15 == 2 )
        {
          sub_AD8380((__int64)&v46, v51);
        }
        else if ( v15 )
        {
          sub_AADB10((__int64)&v46, v17, 1);
        }
        else
        {
          sub_AADB10((__int64)&v46, v17, 0);
        }
        sub_AB2160((__int64)v34, a4, (__int64)&v46, 0);
        sub_969240(&v48);
        sub_969240(&v46);
        if ( v55 )
        {
          v55 = 0;
          sub_22C0090(v50);
        }
        sub_AD8380((__int64)v36, (__int64)v7);
        v18 = v27;
        v19 = v38;
        sub_AD8380((__int64)v38, (__int64)v27);
        v21 = a2[1];
        if ( a6 )
        {
          if ( *(_QWORD *)(v21 + 16) )
          {
            v18 = (_BYTE *)a2[1];
            v19 = v40;
            (*(void (__fastcall **)(__int64 *, _BYTE *, __int64 *, __int64 *))(v21 + 24))(v40, v18, v32, v36);
            v22 = a2[1];
            if ( *(_QWORD *)(v22 + 16) )
            {
              (*(void (__fastcall **)(__int64 *, unsigned __int64, __int64 *, __int64 *))(v22 + 24))(v44, v22, v34, v38);
              sub_AB3510((__int64)&v46, (__int64)v40, (__int64)v44, 0);
              sub_22C06B0((__int64)v50, (__int64)&v46, 0);
              sub_22C0650(a1, v50);
              *(_BYTE *)(a1 + 40) = 1;
              sub_22C0090(v50);
              sub_969240(&v48);
              sub_969240(&v46);
              sub_969240(v45);
              sub_969240(v44);
              sub_969240(&v41);
              sub_969240(v40);
LABEL_24:
              sub_969240(&v39);
              sub_969240(v38);
              sub_969240(&v37);
              sub_969240(v36);
              sub_969240(&v35);
              sub_969240(v34);
              sub_969240(&v33);
              sub_969240(v32);
              return a1;
            }
          }
        }
        else if ( *(_QWORD *)(v21 + 16) )
        {
          v18 = (_BYTE *)a2[1];
          v19 = v42;
          (*(void (__fastcall **)(__int64 *, _BYTE *, __int64 *, __int64 *))(v21 + 24))(v42, v18, v36, v32);
          v24 = a2[1];
          if ( *(_QWORD *)(v24 + 16) )
          {
            (*(void (__fastcall **)(__int64 *, unsigned __int64, __int64 *, __int64 *))(v24 + 24))(v44, v24, v38, v34);
            sub_AB3510((__int64)&v46, (__int64)v42, (__int64)v44, 0);
            sub_22C06B0((__int64)v50, (__int64)&v46, 0);
            sub_22C0650(a1, v50);
            *(_BYTE *)(a1 + 40) = 1;
            sub_22C0090(v50);
            sub_969240(&v48);
            sub_969240(&v46);
            sub_969240(v45);
            sub_969240(v44);
            sub_969240(&v43);
            sub_969240(v42);
            goto LABEL_24;
          }
        }
        sub_4263D6(v19, v18, v20);
      }
    }
  }
  *(_BYTE *)(a1 + 40) = 0;
  return a1;
}
