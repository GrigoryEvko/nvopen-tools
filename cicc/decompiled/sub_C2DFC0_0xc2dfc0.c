// Function: sub_C2DFC0
// Address: 0xc2dfc0
//
unsigned __int64 *__fastcall sub_C2DFC0(unsigned __int64 *a1, __int64 a2, __int64 *a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  _DWORD *v6; // rcx
  __int64 v7; // rsi
  int v8; // eax
  char *v9; // rsi
  unsigned int v10; // ebx
  unsigned __int64 v11; // rax
  size_t v12; // rdx
  char *(*v13)(); // rax
  _BYTE *v15; // rcx
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  char *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rcx
  char *(*v21)(); // rax
  size_t v22; // rdx
  _BYTE *v23; // rcx
  unsigned __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v28; // [rsp+8h] [rbp-198h]
  _BYTE *v29; // [rsp+10h] [rbp-190h] BYREF
  unsigned __int64 v30; // [rsp+18h] [rbp-188h]
  _QWORD v31[2]; // [rsp+20h] [rbp-180h] BYREF
  __int64 v32; // [rsp+30h] [rbp-170h]
  __int64 v33; // [rsp+38h] [rbp-168h]
  __int16 v34; // [rsp+40h] [rbp-160h]
  _QWORD v35[2]; // [rsp+50h] [rbp-150h] BYREF
  const char *v36; // [rsp+60h] [rbp-140h]
  __int16 v37; // [rsp+70h] [rbp-130h]
  _QWORD v38[2]; // [rsp+80h] [rbp-120h] BYREF
  _BYTE *v39; // [rsp+90h] [rbp-110h]
  unsigned __int64 v40; // [rsp+98h] [rbp-108h]
  __int16 v41; // [rsp+A0h] [rbp-100h]
  _QWORD v42[2]; // [rsp+B0h] [rbp-F0h] BYREF
  char *v43; // [rsp+C0h] [rbp-E0h]
  __int16 v44; // [rsp+D0h] [rbp-D0h]
  _BYTE v45[32]; // [rsp+E0h] [rbp-C0h] BYREF
  char v46; // [rsp+100h] [rbp-A0h]
  unsigned int v47; // [rsp+10Ch] [rbp-94h]
  _BYTE *v48; // [rsp+110h] [rbp-90h]
  unsigned __int64 v49; // [rsp+118h] [rbp-88h]
  _BYTE *v50; // [rsp+120h] [rbp-80h] BYREF
  __int64 v51; // [rsp+128h] [rbp-78h]
  _BYTE v52[112]; // [rsp+130h] [rbp-70h] BYREF

  sub_C7C840(v45, a3, 1, 35);
  if ( !v46 )
  {
LABEL_4:
    *a1 = 1;
    return a1;
  }
  while ( 1 )
  {
    v29 = v48;
    v30 = v49;
    v4 = sub_C93580(&v29, 32, 0);
    if ( v4 < v30 )
    {
      v30 -= v4;
      v29 += v4;
      if ( *v29 != 35 )
        break;
    }
LABEL_3:
    sub_C7C5C0(v45);
    if ( !v46 )
      goto LABEL_4;
  }
  v50 = v52;
  v51 = 0x400000000LL;
  sub_C93960(&v29, &v50, 32, 0xFFFFFFFFLL, 0);
  if ( (_DWORD)v51 != 3 )
  {
    v41 = 1283;
    v38[0] = "Expected 'kind mangled_name mangled_name', found '";
    v39 = v29;
    v40 = v30;
    v42[0] = v38;
    v18 = "'";
    goto LABEL_25;
  }
  v5 = *((_QWORD *)v50 + 1);
  v6 = *(_DWORD **)v50;
  if ( v5 == 4 )
  {
    v7 = 0;
    if ( *v6 != 1701667182 )
    {
      if ( *v6 != 1701869940 )
        goto LABEL_18;
      v7 = 1;
    }
LABEL_11:
    v8 = sub_EF7070(a2, v7, *((_QWORD *)v50 + 2), *((_QWORD *)v50 + 3), *((_QWORD *)v50 + 4), *((_QWORD *)v50 + 5));
    switch ( v8 )
    {
      case 2:
        v23 = *(_BYTE **)v50;
        v24 = *((_QWORD *)v50 + 1);
        v34 = 1283;
        v31[0] = "Could not demangle '";
        v32 = *((_QWORD *)v50 + 2);
        v25 = *((_QWORD *)v50 + 3);
        break;
      case 3:
        v23 = *(_BYTE **)v50;
        v24 = *((_QWORD *)v50 + 1);
        v34 = 1283;
        v31[0] = "Could not demangle '";
        v32 = *((_QWORD *)v50 + 4);
        v25 = *((_QWORD *)v50 + 5);
        break;
      case 1:
        v15 = (_BYTE *)*((_QWORD *)v50 + 4);
        v16 = *((_QWORD *)v50 + 5);
        v31[0] = "Manglings '";
        v34 = 1283;
        v32 = *((_QWORD *)v50 + 2);
        v17 = *((_QWORD *)v50 + 3);
        v37 = 770;
        v33 = v17;
        v35[0] = v31;
        v36 = "' and '";
        v39 = v15;
        v40 = v16;
        v41 = 1282;
        v38[0] = v35;
        v42[0] = v38;
        v18 = "' have both been used in prior remappings. Move this remapping earlier in the file.";
LABEL_25:
        v43 = v18;
        v19 = *a3;
        v44 = 770;
        v20 = v47;
        v21 = *(char *(**)())(v19 + 16);
        if ( v21 == sub_C1E8B0 )
        {
          v9 = "Unknown buffer";
          v22 = 14;
        }
        else
        {
          v28 = v47;
          v26 = ((__int64 (__fastcall *)(__int64 *))v21)(a3);
          v20 = v28;
          v9 = (char *)v26;
        }
        sub_C2DD40(a1, v9, v22, v20, (__int64)v42);
        goto LABEL_21;
      default:
        if ( v50 != v52 )
          _libc_free(v50, v7);
        goto LABEL_3;
    }
    v33 = v25;
    v35[0] = v31;
    v36 = "' as a <";
    v38[0] = v35;
    v42[0] = v38;
    v18 = ">; invalid mangling?";
    v37 = 770;
    v39 = v23;
    v40 = v24;
    v41 = 1282;
    goto LABEL_25;
  }
  if ( v5 == 8 && *(_QWORD *)v6 == 0x676E69646F636E65LL )
  {
    v7 = 2;
    goto LABEL_11;
  }
LABEL_18:
  v38[0] = "Invalid kind, expected 'name', 'type', or 'encoding', found '";
  v9 = "Unknown buffer";
  v10 = v47;
  v41 = 1283;
  v39 = *(_BYTE **)v50;
  v11 = *((_QWORD *)v50 + 1);
  v12 = 14;
  v44 = 770;
  v40 = v11;
  v42[0] = v38;
  v43 = "'";
  v13 = *(char *(**)())(*a3 + 16);
  if ( v13 != sub_C1E8B0 )
    v9 = (char *)((__int64 (__fastcall *)(__int64 *, char *, __int64))v13)(a3, "Unknown buffer", 14);
  sub_C2DD40(a1, v9, v12, v10, (__int64)v42);
LABEL_21:
  if ( v50 != v52 )
    _libc_free(v50, v9);
  return a1;
}
