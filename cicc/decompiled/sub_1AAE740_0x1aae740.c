// Function: sub_1AAE740
// Address: 0x1aae740
//
__int64 __fastcall sub_1AAE740(__int64 a1, __int64 *a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r12d
  int v5; // edx
  __int64 v6; // rdi
  unsigned int v7; // ebx
  unsigned int v8; // eax
  unsigned int v10; // r14d
  int v11; // r13d
  int v12; // r13d
  int v13; // r12d
  unsigned int v14; // r14d
  int v15; // r13d
  int v16; // ebx
  unsigned int v17; // r14d
  int v18; // r13d
  unsigned int v19; // r13d
  unsigned int v20; // ebx
  unsigned int v21; // r13d
  int v22; // r14d
  int v23; // r13d
  int v24; // r13d
  int v25; // r13d
  int v26; // r13d
  unsigned int v27; // r13d
  int v28; // r14d
  int v29; // ebx
  unsigned int v30; // r13d
  int v31; // r14d
  unsigned int v32; // r14d
  int v33; // r13d
  unsigned int v34; // ebx
  int v35; // r13d
  unsigned int v36; // r13d
  int v37; // r14d
  unsigned int v38; // r14d
  int v39; // r13d
  unsigned int v40; // r13d
  int v41; // r14d
  int v42; // r13d
  int v43; // ebx
  unsigned int v44; // r13d
  unsigned int v45; // r13d
  int v46; // r14d
  unsigned int v47; // ebx
  unsigned int v48; // r14d
  int v49; // r13d
  unsigned int v50; // r14d
  int v51; // r13d
  unsigned int v52; // r13d
  int v53; // r14d
  int v54; // r13d
  unsigned int v55; // r13d
  int v56; // r14d
  unsigned int v57; // r13d
  int v58; // r14d
  unsigned int v59; // r13d
  unsigned int v60; // ebx
  unsigned int v61; // r14d
  int v62; // r13d
  unsigned int v63; // r13d
  int v64; // r14d
  unsigned int v65; // ebx
  int v66; // r13d
  unsigned int v67; // r14d
  int v68; // r13d
  unsigned int v69; // r13d
  int v70; // r14d
  unsigned int v71; // ebx
  unsigned int v72; // r14d
  int v73; // r13d
  int v74; // r13d
  unsigned int v75; // ebx
  char v76; // [rsp+Eh] [rbp-42h]
  char v77; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v78; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v79; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v80; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v81; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v82; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v83; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v84; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v85; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v86; // [rsp+Fh] [rbp-41h]
  char v87; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v88; // [rsp+Fh] [rbp-41h]
  unsigned __int8 v89; // [rsp+Fh] [rbp-41h]
  char v90; // [rsp+Fh] [rbp-41h]
  int v91; // [rsp+14h] [rbp-3Ch] BYREF
  _QWORD v92[7]; // [rsp+18h] [rbp-38h] BYREF

  LOBYTE(v3) = sub_149CB50(*a2, a1, (unsigned int *)&v91);
  v4 = v3;
  if ( !(_BYTE)v3 )
    return v4;
  v5 = v91;
  if ( (((int)*(unsigned __int8 *)(*a2 + v91 / 4) >> (2 * (v91 & 3))) & 3) == 0 )
    return 0;
  v6 = *(_QWORD *)(a1 + 40);
  v7 = 0;
  if ( v6 )
  {
    LOBYTE(v8) = sub_1633DF0(v6);
    v7 = v8;
    if ( !(_BYTE)v8 )
    {
LABEL_5:
      v5 = v91;
      goto LABEL_6;
    }
    if ( !(unsigned __int8)sub_1560180(a1 + 112, 31) )
    {
      sub_15E0D50(a1, -1, 31);
      goto LABEL_5;
    }
    v5 = v91;
    v7 = 0;
  }
LABEL_6:
  switch ( v5 )
  {
    case 0:
    case 2:
    case 10:
    case 12:
    case 34:
    case 38:
    case 42:
    case 46:
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 32) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 32);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      return v7;
    case 20:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      return v7;
    case 21:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      return v7;
    case 83:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 84:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v65 = sub_1AAE570(a1, 1) | v7;
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v65 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v65 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v65;
    case 94:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      return v7;
    case 97:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 36) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 36);
      }
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      return v7;
    case 114:
    case 116:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 117:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v7;
    case 119:
      v50 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v50 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v51 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v7 | v50 | v51;
    case 141:
    case 142:
    case 143:
    case 144:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_42;
      v4 = 0;
      goto LABEL_43;
    case 145:
      v36 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v36 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v81 = sub_1AAE5D0(a1);
      v37 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
        goto LABEL_534;
      return v37 | v7 | v36 | v81;
    case 146:
      v55 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v55 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v85 = sub_1AAE570(a1, 0);
      v56 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v56 | v7 | v55 | v85;
    case 147:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_387;
      v4 = 0;
      goto LABEL_388;
    case 151:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      return v7;
    case 158:
    case 159:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 160:
    case 161:
    case 171:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      return v7;
    case 187:
    case 189:
    case 191:
    case 195:
    case 196:
    case 200:
    case 202:
    case 227:
    case 232:
    case 233:
    case 235:
    case 240:
    case 241:
    case 243:
    case 244:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_387;
      v4 = 0;
      goto LABEL_388;
    case 188:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      v66 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
        goto LABEL_521;
      return v66 | v7;
    case 190:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      return v7 | (unsigned int)sub_1AAE5D0(a1);
    case 197:
      v38 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v38 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v39 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
        goto LABEL_534;
      return v7 | v38 | v39;
    case 198:
    case 199:
      if ( (unsigned __int8)sub_1560180(a1 + 112, 30) )
        v4 = 0;
      else
        sub_15E0D50(a1, -1, 30);
      v4 |= v7 | sub_1AAE570(a1, 2);
      return v4;
    case 218:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      v74 = sub_1AAE570(a1, 0);
      v90 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( (unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v75 = v74 | v7;
        LOBYTE(v75) = v90 | v75;
      }
      else
      {
        v75 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v75 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v75;
    case 219:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v71 = sub_1AAE570(a1, 1) | v7;
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v71 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v71 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v71;
    case 220:
    case 231:
      v30 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v30 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v80 = sub_1AAE570(a1, 0);
      v31 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
        goto LABEL_521;
      return v31 | v7 | v30 | v80;
    case 221:
    case 222:
    case 228:
    case 229:
    case 230:
    case 236:
    case 238:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      return v7;
    case 223:
    case 224:
      v27 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v27 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v79 = sub_1AAE570(a1, 0);
      v28 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v28 | v7 | v27 | v79;
    case 225:
    case 226:
      if ( (unsigned __int8)sub_1560180(a1 + 112, 30) )
        v4 = 0;
      else
        sub_15E0D50(a1, -1, 30);
      v29 = v4 | sub_1AAE570(a1, 0) | v7;
      return (unsigned int)sub_1AAE570(a1, 3) | v29;
    case 234:
    case 242:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      return v7;
    case 237:
    case 239:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_314;
      v4 = 0;
      goto LABEL_315;
    case 245:
    case 246:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      return (unsigned int)sub_1AAE570(a1, 3) | v7;
    case 247:
    case 248:
    case 253:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_387;
      v4 = 0;
      goto LABEL_388;
    case 249:
    case 250:
    case 255:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      return v7;
    case 251:
      if ( (unsigned __int8)sub_1560180(a1 + 112, 30) )
        v4 = 0;
      else
LABEL_42:
        sub_15E0D50(a1, -1, 30);
LABEL_43:
      v16 = v4 | sub_1AAE5D0(a1) | v7;
      return (unsigned int)sub_1AAE570(a1, 0) | v16;
    case 252:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      return v7;
    case 254:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 256:
      v72 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v72 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v73 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
        goto LABEL_534;
      return v7 | v72 | v73;
    case 257:
    case 258:
    case 304:
    case 305:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 36) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 36);
      }
      return v7;
    case 263:
      v67 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v67 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v68 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v7 | v67 | v68;
    case 283:
      v57 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v57 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v86 = sub_1AAE570(a1, 0);
      v58 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v58 | v7 | v57 | v86;
    case 284:
    case 358:
    case 360:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 285:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      return v7;
    case 286:
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      return v7;
    case 287:
    case 290:
    case 291:
    case 292:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v7;
    case 288:
    case 293:
      v24 = sub_1AAE5D0(a1);
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_518;
      return v24 | v7;
    case 289:
      v43 = sub_1AAE5D0(a1) | v7;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v43 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v44 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( (unsigned __int8)sub_1560290(v92, 1, 22) )
        return v43 | v44;
      else
LABEL_534:
        sub_15E0DF0(a1, 1, 22);
      return v4;
    case 295:
      v45 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 4) )
      {
        v45 = v4;
        sub_15E0D50(a1, -1, 4);
      }
      v83 = sub_1AAE570(a1, 0);
      v46 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
        goto LABEL_521;
      return v46 | v7 | v45 | v83;
    case 296:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 297:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_387;
      v4 = 0;
      goto LABEL_388;
    case 298:
    case 299:
    case 300:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      return v7;
    case 306:
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 307:
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 308:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 309:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_387;
      v4 = 0;
      goto LABEL_388;
    case 310:
    case 317:
    case 322:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 311:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      v47 = sub_1AAE570(a1, 0) | v7;
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v47 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v47 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v47 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v47;
    case 316:
    case 325:
      return (unsigned int)sub_1AAE570(a1, 1) | v7;
    case 318:
    case 319:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_314;
      v4 = 0;
      goto LABEL_315;
    case 320:
    case 321:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      return v7;
    case 323:
      v54 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
        goto LABEL_521;
      return v54 | v7;
    case 324:
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 3, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 3, 22);
      }
      return v7;
    case 326:
      v63 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v63 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v88 = sub_1AAE570(a1, 0);
      v64 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v64 | v7 | v63 | v88;
    case 327:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      return v7 | (unsigned int)sub_1AAE570(a1, 0);
    case 329:
    case 330:
    case 336:
      v17 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v17 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v18 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v7 | v17 | v18;
    case 331:
      v59 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v59 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v87 = sub_1AAE570(a1, 0);
      v76 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( (unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        LOBYTE(v59) = v87 | v59;
        v60 = v59 | v7;
        LOBYTE(v60) = v76 | v60;
      }
      else
      {
        v60 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v60 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v60;
    case 332:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_387;
      v4 = 0;
      goto LABEL_388;
    case 340:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 341:
    case 343:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_387;
      v4 = 0;
      goto LABEL_388;
    case 342:
      v69 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v69 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v89 = sub_1AAE570(a1, 1);
      v70 = sub_1AAE570(a1, 2);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
        goto LABEL_521;
      return v70 | v7 | v69 | v89;
    case 351:
      v40 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v40 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v82 = sub_1AAE570(a1, 0);
      v41 = sub_1AAE570(a1, 2);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( (unsigned __int8)sub_1560290(v92, 2, 37) )
        return v41 | v7 | v40 | v82;
      else
        sub_15E0DF0(a1, 2, 37);
      return v4;
    case 352:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v42 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
        goto LABEL_521;
      return v7 | v42;
    case 356:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v7;
    case 357:
    case 359:
      v21 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v21 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v78 = sub_1AAE570(a1, 0);
      v22 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v22 | v7 | v21 | v78;
    case 361:
    case 362:
    case 364:
    case 368:
    case 373:
    case 375:
      v14 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v14 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v15 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
        goto LABEL_521;
      return v7 | v14 | v15;
    case 363:
    case 366:
    case 367:
    case 369:
    case 372:
    case 374:
    case 380:
      v12 = sub_1AAE5D0(a1);
      if ( (unsigned __int8)sub_1560180(a1 + 112, 30) )
        v4 = v12 | v7;
      else
        sub_15E0D50(a1, -1, 30);
      v13 = sub_1AAE570(a1, 0) | v4;
      return (unsigned int)sub_1AAE570(a1, 1) | v13;
    case 365:
    case 379:
      v23 = sub_1AAE5D0(a1);
      if ( (unsigned __int8)sub_1560180(a1 + 112, 30) )
        return v23 | v7;
      else
LABEL_518:
        sub_15E0D50(a1, -1, 30);
      return v4;
    case 370:
    case 376:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      v25 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v7 | v25;
    case 371:
    case 420:
      v7 |= sub_1AAE5D0(a1);
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 4) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 4);
      }
      return v7 | (unsigned int)sub_1AAE570(a1, 0);
    case 378:
    case 381:
      v26 = sub_1AAE5D0(a1);
      if ( (unsigned __int8)sub_1560180(a1 + 112, 30) )
        v4 = v26 | v7;
      else
        sub_15E0D50(a1, -1, 30);
      v4 |= sub_1AAE570(a1, 1);
      return v4;
    case 382:
    case 383:
    case 386:
    case 387:
    case 388:
    case 389:
    case 390:
      v10 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v10 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v11 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      return v7 | v10 | v11;
    case 384:
    case 385:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v7;
    case 391:
      v52 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v52 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v84 = sub_1AAE570(a1, 0);
      v53 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
        goto LABEL_521;
      return v53 | v7 | v52 | v84;
    case 392:
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 399:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
        goto LABEL_387;
      v4 = 0;
      goto LABEL_388;
    case 400:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      return v7;
    case 401:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      return v7;
    case 406:
      if ( (unsigned __int8)sub_1560180(a1 + 112, 30) )
        v4 = 0;
      else
LABEL_387:
        sub_15E0D50(a1, -1, 30);
LABEL_388:
      v4 |= v7 | sub_1AAE570(a1, 0);
      return v4;
    case 407:
      if ( (unsigned __int8)sub_1560180(a1 + 112, 30) )
        v4 = 0;
      else
LABEL_314:
        sub_15E0D50(a1, -1, 30);
LABEL_315:
      v4 |= v7 | sub_1AAE570(a1, 1);
      return v4;
    case 408:
      v61 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v61 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v62 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      v4 = v7 | v61 | v62;
      break;
    case 409:
      v48 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v48 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v49 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
        goto LABEL_526;
      v4 = v7 | v48 | v49;
      break;
    case 410:
    case 411:
      v19 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v19 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v77 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( (unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        LOBYTE(v19) = v77 | v19;
        v20 = v19 | v7;
      }
      else
      {
        v20 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v20 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v20 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v20;
    case 412:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560260(v92, 0, 20) )
      {
        v7 = v4;
        sub_15E0D50(a1, 0, 20);
      }
      return v7;
    case 413:
    case 418:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v7;
    case 414:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      return v7;
    case 415:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      return v7;
    case 416:
      v32 = 0;
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v32 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v33 = sub_1AAE570(a1, 0);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( (unsigned __int8)sub_1560290(v92, 0, 37) )
        v4 = v7 | v32 | v33;
      else
LABEL_526:
        sub_15E0DF0(a1, 0, 37);
      break;
    case 417:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 2, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 2, 22);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 2, 37) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 2, 37);
      }
      v4 = v7;
      break;
    case 419:
      if ( !(unsigned __int8)sub_1560180(a1 + 112, 30) )
      {
        v7 = v4;
        sub_15E0D50(a1, -1, 30);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 22) )
      {
        v7 = v4;
        sub_15E0DF0(a1, 0, 22);
      }
      v34 = sub_1AAE570(a1, 1) | v7;
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 0, 37) )
      {
        v34 = v4;
        sub_15E0DF0(a1, 0, 37);
      }
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( !(unsigned __int8)sub_1560290(v92, 1, 37) )
      {
        v34 = v4;
        sub_15E0DF0(a1, 1, 37);
      }
      v4 = v34;
      break;
    case 421:
      v35 = sub_1AAE570(a1, 1);
      v92[0] = *(_QWORD *)(a1 + 112);
      if ( (unsigned __int8)sub_1560290(v92, 1, 37) )
        v4 = v35 | v7;
      else
LABEL_521:
        sub_15E0DF0(a1, 1, 37);
      break;
    default:
      v4 = 0;
      break;
  }
  return v4;
}
