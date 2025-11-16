// Function: sub_149B780
// Address: 0x149b780
//
char __fastcall sub_149B780(__int64 a1, __int64 *a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r13
  int v8; // edx
  __int64 v9; // rax
  char result; // al
  _QWORD *v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rax
  bool v16; // cl
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // rcx
  __int64 v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rdx
  _QWORD *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rax
  bool v28; // cl
  _QWORD *v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rax
  bool v33; // cl
  _QWORD *v34; // rax
  __int64 v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  _QWORD *v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rdx
  _QWORD *v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  _QWORD *v50; // rdx
  _QWORD *v51; // rdx
  _QWORD *v52; // rax
  __int64 v53; // rax
  _QWORD *v54; // rcx
  __int64 v55; // rax
  bool v56; // cl
  _QWORD *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  _QWORD *v60; // rax
  _QWORD *v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  _QWORD *v64; // rax
  __int64 v65; // rax
  _QWORD *v66; // rax
  _QWORD *v67; // rax
  _QWORD *v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  _QWORD *v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // rdx
  __int64 v76; // rax
  _QWORD *v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rax
  _QWORD *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  _QWORD *v83; // rax
  _QWORD *v84; // rax
  _QWORD *v85; // rdx
  __int64 v86; // rax
  _QWORD *v87; // rdx
  __int64 v88; // rax
  _QWORD *v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rax
  _QWORD *v96; // rax
  __int64 v97; // rax

  v5 = a3;
  v6 = *a2;
  v7 = sub_16471D0(*a2, 0);
  if ( a4 )
    a4 = sub_15A9620(a4, v6, 0);
  v8 = *((_DWORD *)a2 + 3);
  v9 = (unsigned int)(v8 - 1);
  switch ( v5 )
  {
    case 0LL:
    case 2LL:
    case 10LL:
    case 12LL:
    case 34LL:
    case 38LL:
    case 42LL:
    case 46LL:
      if ( v8 != 2 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 15;
    case 1LL:
    case 3LL:
    case 11LL:
    case 13LL:
    case 35LL:
    case 36LL:
    case 39LL:
    case 40LL:
    case 43LL:
    case 44LL:
    case 47LL:
    case 48LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 15;
    case 4LL:
    case 7LL:
    case 14LL:
    case 17LL:
    case 22LL:
    case 28LL:
      if ( v8 != 2 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 5LL:
    case 6LL:
    case 8LL:
    case 9LL:
    case 15LL:
    case 16LL:
    case 18LL:
    case 19LL:
    case 23LL:
    case 24LL:
    case 26LL:
    case 27LL:
    case 29LL:
    case 30LL:
    case 32LL:
    case 33LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 20LL:
    case 141LL:
    case 142LL:
    case 143LL:
    case 144LL:
    case 190LL:
    case 251LL:
    case 254LL:
    case 259LL:
    case 309LL:
    case 310LL:
    case 317LL:
    case 322LL:
    case 406LL:
    case 408LL:
    case 409LL:
      if ( v8 != 2 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 21LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 25LL:
    case 31LL:
      if ( v8 != 4 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 37LL:
    case 41LL:
    case 45LL:
    case 49LL:
      if ( v8 != 4 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 15;
    case 50LL:
    case 51LL:
    case 52LL:
    case 53LL:
    case 54LL:
    case 55LL:
    case 56LL:
    case 57LL:
    case 58LL:
    case 62LL:
    case 63LL:
    case 64LL:
    case 65LL:
    case 66LL:
    case 67LL:
    case 74LL:
    case 75LL:
    case 76LL:
    case 77LL:
    case 78LL:
    case 79LL:
    case 80LL:
    case 81LL:
    case 82LL:
    case 85LL:
    case 86LL:
    case 87LL:
    case 88LL:
    case 89LL:
    case 90LL:
    case 91LL:
    case 92LL:
    case 93LL:
    case 103LL:
    case 104LL:
    case 105LL:
    case 108LL:
    case 109LL:
    case 110LL:
    case 120LL:
    case 121LL:
    case 122LL:
    case 123LL:
    case 124LL:
    case 125LL:
    case 126LL:
    case 127LL:
    case 128LL:
    case 129LL:
    case 130LL:
    case 131LL:
    case 132LL:
    case 136LL:
    case 137LL:
    case 138LL:
    case 139LL:
    case 140LL:
    case 152LL:
    case 153LL:
    case 154LL:
    case 155LL:
    case 156LL:
    case 157LL:
    case 165LL:
    case 166LL:
    case 167LL:
    case 168LL:
    case 169LL:
    case 170LL:
    case 172LL:
    case 173LL:
    case 174LL:
    case 175LL:
    case 176LL:
    case 177LL:
    case 178LL:
    case 179LL:
    case 180LL:
    case 181LL:
    case 182LL:
    case 183LL:
    case 184LL:
    case 185LL:
    case 186LL:
    case 203LL:
    case 204LL:
    case 205LL:
    case 268LL:
    case 269LL:
    case 270LL:
    case 271LL:
    case 272LL:
    case 273LL:
    case 274LL:
    case 275LL:
    case 276LL:
    case 277LL:
    case 278LL:
    case 279LL:
    case 280LL:
    case 281LL:
    case 282LL:
    case 301LL:
    case 302LL:
    case 303LL:
    case 333LL:
    case 334LL:
    case 335LL:
    case 337LL:
    case 338LL:
    case 339LL:
    case 344LL:
    case 345LL:
    case 346LL:
    case 347LL:
    case 348LL:
    case 349LL:
    case 353LL:
    case 354LL:
    case 355LL:
    case 393LL:
    case 394LL:
    case 395LL:
    case 396LL:
    case 397LL:
    case 398LL:
    case 403LL:
    case 404LL:
    case 405LL:
      if ( v8 != 2 )
        return 0;
      v54 = (_QWORD *)a2[2];
      if ( (unsigned __int8)(*(_BYTE *)(*v54 + 8LL) - 1) > 5u )
        return 0;
      return v54[1] == *v54;
    case 59LL:
    case 60LL:
    case 61LL:
    case 98LL:
    case 99LL:
    case 100LL:
    case 133LL:
    case 134LL:
    case 135LL:
    case 162LL:
    case 163LL:
    case 164LL:
    case 209LL:
    case 210LL:
    case 211LL:
    case 212LL:
    case 213LL:
    case 214LL:
    case 215LL:
    case 216LL:
    case 217LL:
    case 313LL:
    case 314LL:
    case 315LL:
      if ( v8 != 3 )
        return 0;
      v11 = (_QWORD *)a2[2];
      v12 = *v11;
      if ( (unsigned __int8)(*(_BYTE *)(*v11 + 8LL) - 1) > 5u || v12 != v11[1] )
        return 0;
      return v11[2] == v12;
    case 68LL:
    case 106LL:
      if ( v8 != 2 )
        return 0;
      v50 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v50 + 8LL) != 3 )
        return 0;
      return v50[1] == *v50;
    case 69LL:
    case 107LL:
      if ( v8 != 2 )
        return 0;
      v51 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v51 + 8LL) != 2 )
        return 0;
      return v51[1] == *v51;
    case 70LL:
      if ( v8 != 4 )
        return 0;
      v61 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v61 + 8LL) != 11 || *(_BYTE *)(v61[1] + 8LL) != 15 || *(_BYTE *)(v61[2] + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v61[3] + 8LL) == 15;
    case 71LL:
    case 72LL:
    case 73LL:
    case 97LL:
      if ( v8 != 2 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 83LL:
      if ( v8 == 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 84LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      v62 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v62 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v62 + 16) + 8LL) == 15;
    case 94LL:
    case 95LL:
      v15 = *(_QWORD *)(a2[2] + 8 * v9);
      if ( a4 )
        v16 = a4 == v15;
      else
        v16 = *(_BYTE *)(v15 + 8) == 11;
      LODWORD(v9) = v8 - 2;
      if ( !v16 )
        return 0;
      goto LABEL_48;
    case 96LL:
      v55 = *(_QWORD *)(a2[2] + 8 * v9);
      if ( a4 )
        v56 = a4 == v55;
      else
        v56 = *(_BYTE *)(v55 + 8) == 11;
      LODWORD(v9) = v8 - 2;
      if ( v56 )
        goto LABEL_175;
      return 0;
    case 101LL:
    case 102LL:
      if ( v8 != 2 )
        return 0;
      return (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) - 1) <= 5u;
    case 111LL:
    case 113LL:
      v32 = *(_QWORD *)(a2[2] + 8 * v9);
      if ( a4 )
        v33 = a4 == v32;
      else
        v33 = *(_BYTE *)(v32 + 8) == 11;
      LODWORD(v9) = v8 - 2;
      if ( v33 )
        goto LABEL_92;
      return 0;
    case 112LL:
    case 115LL:
      v27 = *(_QWORD *)(a2[2] + 8 * v9);
      if ( a4 )
        v28 = a4 == v27;
      else
        v28 = *(_BYTE *)(v27 + 8) == 11;
      LODWORD(v9) = v8 - 2;
      if ( v28 )
        goto LABEL_83;
      return 0;
    case 114LL:
    case 116LL:
      if ( v8 == 1 )
        return 0;
      v44 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v44 + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v44[1] + 8LL) == 15;
    case 117LL:
      if ( v8 != 4 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 118LL:
    case 262LL:
    case 267LL:
      if ( v8 != 2 )
        return 0;
      v25 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v25 + 8LL) != 11 )
        return 0;
      return v25[1] == *v25;
    case 119LL:
    case 158LL:
    case 159LL:
    case 160LL:
    case 161LL:
    case 171LL:
    case 187LL:
    case 189LL:
    case 191LL:
    case 195LL:
    case 196LL:
    case 200LL:
    case 202LL:
    case 227LL:
    case 232LL:
    case 233LL:
    case 234LL:
    case 235LL:
    case 240LL:
    case 241LL:
    case 242LL:
    case 243LL:
    case 244LL:
    case 247LL:
    case 248LL:
    case 253LL:
    case 296LL:
    case 297LL:
    case 399LL:
      if ( v8 == 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 145LL:
    case 146LL:
      if ( v8 != 4 )
        return 0;
      v38 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v38 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v38 + 16) + 8LL) == 15;
    case 147LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 148LL:
    case 149LL:
    case 150LL:
      v20 = (_QWORD *)a2[2];
      v21 = *v20;
      if ( (unsigned __int8)(*(_BYTE *)(*v20 + 8LL) - 1) > 5u )
        return 0;
      if ( v8 == 2 )
      {
        v97 = v20[1];
        if ( *(_BYTE *)(v97 + 8) == 14 && *(_QWORD *)(v97 + 32) == 2 )
          return **(_QWORD **)(v97 + 16) == v21;
      }
      else if ( v8 == 3 && v21 == v20[1] )
      {
        return v20[2] == v21;
      }
      return 0;
    case 151LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 15;
    case 188LL:
      if ( v8 != 3 )
        return 0;
      v60 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v60 + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v60[2] + 8LL) == 15;
    case 192LL:
    case 193LL:
    case 194LL:
    case 206LL:
    case 207LL:
    case 208LL:
      if ( v8 != 2 || !(unsigned __int8)sub_1642F90(*(_QWORD *)a2[2], 32) )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 11;
    case 197LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      v69 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v69 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v69 + 16) + 8LL) == 15;
    case 198LL:
    case 199LL:
      if ( v8 != 4 )
        return 0;
      v39 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v39 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v39 + 24) + 8LL) == 15;
    case 201LL:
    case 220LL:
    case 231LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      v22 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v22 + 8LL) != 11 || *(_BYTE *)(v22[1] + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v22[2] + 8LL) == 15;
    case 218LL:
      if ( v8 != 3 )
        return 0;
      v66 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v66 + 8LL) != 15 || *(_BYTE *)(v66[1] + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v66[2] + 8LL) == 15;
    case 219LL:
      if ( v8 != 3 )
        return 0;
      v68 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v68 + 8LL) != 15 || *(_BYTE *)(v68[1] + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v68[2] + 8LL) == 15;
    case 221LL:
    case 222LL:
    case 228LL:
    case 229LL:
    case 230LL:
    case 236LL:
    case 238LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 223LL:
    case 224LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      v49 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v49 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v49 + 16) + 8LL) == 15;
    case 225LL:
    case 226LL:
      if ( v8 != 5 )
        return 0;
      v48 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v48 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v48 + 32) + 8LL) == 15;
    case 237LL:
    case 239LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 245LL:
    case 246LL:
      if ( v8 != 5 )
        return 0;
      v47 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v47 + 8LL) != 11
        || *(_BYTE *)(v47[1] + 8LL) != 15
        || *(_BYTE *)(v47[2] + 8LL) != 11
        || *(_BYTE *)(v47[3] + 8LL) != 11 )
      {
        return 0;
      }
      return *(_BYTE *)(v47[4] + 8LL) == 15;
    case 249LL:
    case 250LL:
      if ( v8 != 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 11;
    case 252LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 255LL:
      if ( v8 != 2 )
        return 0;
      return *(_QWORD *)(a2[2] + 8) == v7;
    case 256LL:
      if ( v8 != 3 )
        return 0;
      v92 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v92 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v92 + 16) + 8LL) == 15;
    case 257LL:
    case 304LL:
      if ( v8 != 2 || !(unsigned __int8)sub_1642F90(*(_QWORD *)a2[2], 32) )
        return 0;
      return *(_QWORD *)(a2[2] + 8) == *(_QWORD *)a2[2];
    case 258LL:
    case 305LL:
      if ( v8 != 2 || !(unsigned __int8)sub_1642F90(*(_QWORD *)a2[2], 16) )
        return 0;
      return *(_QWORD *)(a2[2] + 8) == *(_QWORD *)a2[2];
    case 260LL:
    case 261LL:
    case 320LL:
    case 321LL:
    case 402LL:
      if ( v8 != 2 || !(unsigned __int8)sub_1642F90(*(_QWORD *)a2[2], 32) )
        return 0;
      return *(_QWORD *)(a2[2] + 8) == *(_QWORD *)a2[2];
    case 263LL:
      if ( v8 != 4 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 264LL:
    case 265LL:
    case 266LL:
      if ( v8 != 3 )
        return 0;
      v24 = (_QWORD *)a2[2];
      if ( (unsigned __int8)(*(_BYTE *)(*v24 + 8LL) - 1) > 5u || *v24 != v24[1] )
        return 0;
      return sub_1642F90(v24[2], 32);
    case 283LL:
      if ( v8 != 3 )
        return 0;
      v65 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v65 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v65 + 16) + 8LL) == 15;
    case 284LL:
    case 358LL:
    case 360LL:
      if ( v8 != 3 )
        return 0;
      v26 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v26 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v26 + 16) + 8LL) == 15;
    case 285LL:
      if ( v8 != 2 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 15;
    case 286LL:
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 15;
    case 287LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 288LL:
    case 293LL:
      if ( v8 != 4 )
        return 0;
      v36 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v36 + 8LL) != 15 || *v36 != v36[1] || !(unsigned __int8)sub_1642F90(v36[2], 32) )
        return 0;
      v37 = *(_QWORD *)(a2[2] + 24);
      if ( a4 )
        return a4 == v37;
      else
        return *(_BYTE *)(v37 + 8) == 11;
    case 289LL:
      if ( v8 != 4 )
        return 0;
      if ( !(unsigned __int8)sub_1642F90(*(_QWORD *)a2[2], 32) )
        return 0;
      v91 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v91 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v91 + 16) + 8LL) == 15;
    case 290LL:
    case 291LL:
    case 292LL:
LABEL_48:
      if ( (_DWORD)v9 != 3 )
        return 0;
      v17 = (_QWORD *)a2[2];
      v18 = v17[1];
      if ( v18 != *v17 || *(_BYTE *)(v18 + 8) != 15 || *(_BYTE *)(v17[2] + 8LL) != 15 )
        return 0;
      v19 = v17[3];
      if ( a4 )
        return a4 == v19;
      else
        return *(_BYTE *)(v19 + 8) == 11;
    case 294LL:
LABEL_175:
      if ( (_DWORD)v9 != 3 )
        return 0;
      v57 = (_QWORD *)a2[2];
      v58 = v57[1];
      if ( v58 != *v57 || *(_BYTE *)(v58 + 8) != 15 || *(_BYTE *)(v57[2] + 8LL) != 11 )
        return 0;
      v59 = v57[3];
      if ( a4 )
        return a4 == v59;
      else
        return *(_BYTE *)(v59 + 8) == 11;
    case 295LL:
      if ( *((_DWORD *)a2 + 2) >> 8 )
        return 0;
      if ( v8 != 4 )
        return 0;
      v96 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(v96[1] + 8LL) != 15 || *(_BYTE *)(v96[2] + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v96[3] + 8LL) == 11;
    case 298LL:
    case 299LL:
    case 300LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 306LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 307LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 308LL:
      if ( v8 != 2 )
        return 0;
      v84 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v84 + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v84[1] + 8LL) == 15;
    case 311LL:
      if ( v8 != 3 )
        return 0;
      v83 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v83 + 8LL) != 15 || *(_BYTE *)(v83[1] + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v83[2] + 8LL) == 15;
    case 312LL:
      if ( v8 != 4 )
        return 0;
      if ( !(unsigned __int8)sub_1642F90(*(_QWORD *)a2[2], 32) )
        return 0;
      v64 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(v64[1] + 8LL) != 15 || a4 != v64[2] )
        return 0;
      return v64[3] == a4;
    case 316LL:
    case 323LL:
      if ( v8 != 5 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 318LL:
    case 319LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 324LL:
      if ( v8 != 5 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 32) + 8LL) == 15;
    case 325LL:
      if ( v8 != 4 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 326LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      v63 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v63 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v63 + 16) + 8LL) == 15;
    case 327LL:
    case 328LL:
      if ( v8 != 3 )
        return 0;
      v52 = (_QWORD *)a2[2];
      if ( v7 != *v52 || v7 != v52[1] )
        return 0;
      v53 = v52[2];
      if ( a4 )
        return a4 == v53;
      else
        return *(_BYTE *)(v53 + 8) == 11;
    case 329LL:
    case 330LL:
    case 332LL:
    case 336LL:
      if ( v8 == 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 331LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      v95 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v95 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v95 + 16) + 8LL) == 15;
    case 340LL:
    case 341LL:
    case 343LL:
      if ( v8 == 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 342LL:
      if ( v8 != 4 )
        return 0;
      v70 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v70 + 16) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v70 + 24) + 8LL) == 15;
    case 350LL:
    case 352LL:
    case 356LL:
    case 357LL:
    case 359LL:
      if ( (unsigned int)v9 > 1 )
      {
        v14 = (_QWORD *)a2[2];
        if ( *(_BYTE *)(v14[1] + 8LL) == 15 && *(_BYTE *)(v14[2] + 8LL) == 15 )
          return sub_1642F90(*v14, 32);
      }
      return 0;
    case 351LL:
      if ( v8 != 4 )
        return 0;
      v14 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(v14[1] + 8LL) != 15 )
        return 0;
      if ( *(_BYTE *)(v14[3] + 8LL) == 15 )
        return sub_1642F90(*v14, 32);
      else
        return 0;
    case 361LL:
    case 368LL:
LABEL_92:
      if ( (_DWORD)v9 != 2 )
        return 0;
      v34 = (_QWORD *)a2[2];
      v35 = v34[1];
      if ( *v34 != v35 )
        return 0;
      return v7 == v35 && v34[2] == v35;
    case 362LL:
    case 375LL:
LABEL_83:
      if ( (_DWORD)v9 != 3 )
        return 0;
      v29 = (_QWORD *)a2[2];
      v30 = v29[1];
      if ( *v29 != v30 )
        return 0;
      result = v7 == v30 && v29[2] == v30;
      if ( !result )
        return 0;
      v31 = v29[3];
      if ( a4 )
      {
        if ( a4 != v31 )
          return 0;
      }
      else if ( *(_BYTE *)(v31 + 8) != 11 )
      {
        return 0;
      }
      return result;
    case 363LL:
    case 367LL:
    case 372LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      v23 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v23 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v23 + 16) + 8LL) == 15;
    case 364LL:
      if ( v8 != 3 )
        return 0;
      v85 = (_QWORD *)a2[2];
      v86 = *v85;
      if ( *(_BYTE *)(*v85 + 8LL) != 15 || v86 != v85[1] )
        return 0;
      return v85[2] == v86;
    case 365LL:
    case 379LL:
      if ( v8 != 3 )
        return 0;
      v43 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v43 + 8LL) != 15 || *v43 != v43[1] )
        return 0;
      return *(_BYTE *)(v43[2] + 8LL) == 11;
    case 366LL:
      if ( v8 != 3 )
        return 0;
      if ( !(unsigned __int8)sub_1642F90(*(_QWORD *)a2[2], 32) )
        return 0;
      v81 = a2[2];
      v82 = *(_QWORD *)(v81 + 8);
      if ( *(_BYTE *)(v82 + 8) != 15 )
        return 0;
      return *(_QWORD *)(v81 + 16) == v82;
    case 369LL:
    case 380LL:
      if ( v8 != 3 )
        return 0;
      v41 = (_QWORD *)a2[2];
      v42 = v41[1];
      if ( *(_BYTE *)(v42 + 8) != 15 || v41[2] != v42 )
        return 0;
      return *(_BYTE *)(*v41 + 8LL) == 11;
    case 370LL:
    case 376LL:
      if ( v8 == 1 )
        return 0;
      v40 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v40 + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v40[1] + 8LL) == 15;
    case 371LL:
      if ( v8 != 2 )
        return 0;
      v80 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(v80[1] + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*v80 + 8LL) == 11;
    case 373LL:
      if ( v8 != 4 )
        return 0;
      v77 = (_QWORD *)a2[2];
      v78 = *v77;
      if ( *(_BYTE *)(*v77 + 8LL) != 15 || v78 != v77[1] || v78 != v77[2] )
        return 0;
      v79 = v77[3];
      if ( a4 )
        return a4 == v79;
      else
        return *(_BYTE *)(v79 + 8) == 11;
    case 374LL:
      if ( v8 != 4 )
        return 0;
      if ( !(unsigned __int8)sub_1642F90(*(_QWORD *)a2[2], 32) )
        return 0;
      v74 = (_QWORD *)a2[2];
      v75 = v74[1];
      if ( *(_BYTE *)(v75 + 8) != 15 || v75 != v74[2] )
        return 0;
      v76 = v74[3];
      if ( a4 )
        return a4 == v76;
      else
        return *(_BYTE *)(v76 + 8) == 11;
    case 377LL:
      if ( v8 != 3 )
        return 0;
      v89 = (_QWORD *)a2[2];
      v90 = v89[2];
      if ( v90 != *v89 )
        return 0;
      return a4 == v90 && v89[1] == v7;
    case 378LL:
      if ( v8 != 3 )
        return 0;
      v87 = (_QWORD *)a2[2];
      v88 = v87[1];
      if ( *(_BYTE *)(v88 + 8) != 15 || *v87 != v88 )
        return 0;
      return v87[2] == v88;
    case 381LL:
      if ( v8 != 3 )
        return 0;
      v67 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(*v67 + 8LL) != 15 || *(_BYTE *)(v67[1] + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(v67[2] + 8LL) == 15;
    case 382LL:
    case 383LL:
    case 386LL:
    case 387LL:
    case 388LL:
    case 389LL:
    case 390LL:
      if ( (unsigned int)(v8 - 3) > 1 )
        return 0;
      v13 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v13 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v13 + 16) + 8LL) == 15;
    case 384LL:
    case 385LL:
      if ( (unsigned int)v9 <= 1 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 391LL:
      if ( v8 != 4 )
        return 0;
      v93 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v93 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v93 + 16) + 8LL) == 15;
    case 392LL:
      if ( v8 != 2 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 400LL:
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 15;
    case 401LL:
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 15;
    case 407LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 410LL:
    case 411LL:
      if ( v8 != 3 )
        return 0;
      v46 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v46 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v46 + 16) + 8LL) == 15;
    case 412LL:
      return *(_BYTE *)(*(_QWORD *)a2[2] + 8LL) == 15;
    case 413LL:
    case 418LL:
      if ( v8 != 4 )
        return 0;
      v45 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v45 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v45 + 16) + 8LL) == 15;
    case 414LL:
      if ( v8 != 4 )
        return 0;
      v94 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v94 + 16) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v94 + 24) + 8LL) == 15;
    case 415LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 8) + 8LL) == 15;
    case 416LL:
      if ( v8 != 3 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
    case 417LL:
      if ( v8 != 5 )
        return 0;
      v73 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v73 + 8) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v73 + 24) + 8LL) == 15;
    case 419LL:
      if ( v8 != 4 )
        return 0;
      v72 = a2[2];
      if ( *(_BYTE *)(*(_QWORD *)(v72 + 16) + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(v72 + 24) + 8LL) == 15;
    case 420LL:
      if ( v8 != 2 )
        return 0;
      v71 = (_QWORD *)a2[2];
      if ( *(_BYTE *)(v71[1] + 8LL) != 15 )
        return 0;
      return *(_BYTE *)(*v71 + 8LL) == 11;
    case 421LL:
      if ( v8 != 4 )
        return 0;
      return *(_BYTE *)(*(_QWORD *)(a2[2] + 16) + 8LL) == 15;
  }
}
