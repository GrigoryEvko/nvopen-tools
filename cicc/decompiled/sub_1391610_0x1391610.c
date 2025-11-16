// Function: sub_1391610
// Address: 0x1391610
//
__int64 __fastcall sub_1391610(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 result; // rax
  unsigned __int8 v9; // al
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // r13
  char v16; // cl
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rcx
  unsigned __int8 v22; // al
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // r13
  unsigned __int8 v30; // al
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int8 v36; // dl
  unsigned __int8 v37; // al
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rdi
  unsigned __int8 v43; // al
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rdx
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 v51; // rax
  __int64 v52; // r15
  __int64 v53; // rax
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // [rsp+8h] [rbp-38h]

  v4 = a1;
  v5 = a2;
  while ( 2 )
  {
    switch ( *(_WORD *)(v5 + 18) )
    {
      case 0xB:
      case 0xD:
      case 0xE:
      case 0xF:
      case 0x10:
      case 0x11:
      case 0x12:
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
      case 0x17:
      case 0x18:
      case 0x19:
      case 0x1A:
      case 0x1B:
      case 0x1C:
      case 0x33:
      case 0x34:
      case 0x3D:
        v6 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
        v7 = *(_QWORD *)(v5 - 24 * v6);
        if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 15 )
          goto LABEL_11;
        result = *(_QWORD *)v5;
        if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 15 )
        {
          v9 = *(_BYTE *)(v7 + 16);
          if ( v9 > 3u )
          {
            if ( v9 == 5 )
            {
              if ( (unsigned int)*(unsigned __int16 *)(v7 + 18) - 51 > 1
                && (unsigned __int8)sub_13848E0(
                                      *(_QWORD *)(v4 + 24),
                                      *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)),
                                      0,
                                      0) )
              {
                sub_1391610(v4, v7);
              }
            }
            else
            {
              sub_13848E0(*(_QWORD *)(v4 + 24), *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)), 0, 0);
            }
          }
          else
          {
            v10 = *(_QWORD *)(v4 + 24);
            v11 = sub_14C81A0(*(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)));
            v12 = v10;
            if ( (unsigned __int8)sub_13848E0(v10, v7, 0, v11) )
            {
              v50 = *(_QWORD *)(v4 + 24);
              v51 = sub_14C8160(v12, v7, v13);
              sub_13848E0(v50, v7, 1u, v51);
            }
          }
          if ( v5 != v7 )
            sub_1391C50(v4, v7, v5, 0);
          v6 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
LABEL_11:
          v14 = 1;
          goto LABEL_32;
        }
        return result;
      case 0xC:
      case 0x1D:
      case 0x1E:
      case 0x1F:
      case 0x21:
      case 0x22:
      case 0x23:
      case 0x31:
      case 0x32:
      case 0x35:
      case 0x36:
      case 0x38:
      case 0x39:
      case 0x3A:
      case 0x3C:
      case 0x3F:
        v19 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
        v20 = *(_QWORD *)(v5 - 24 * v19);
        if ( *(_BYTE *)(*(_QWORD *)v20 + 8LL) == 15 && *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 15 )
        {
          v43 = *(_BYTE *)(v20 + 16);
          if ( v43 > 3u )
          {
            if ( v43 == 5 )
            {
              if ( (unsigned int)*(unsigned __int16 *)(v20 + 18) - 51 > 1
                && (unsigned __int8)sub_13848E0(
                                      *(_QWORD *)(v4 + 24),
                                      *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)),
                                      0,
                                      0) )
              {
                sub_1391610(v4, v20);
              }
            }
            else
            {
              sub_13848E0(*(_QWORD *)(v4 + 24), *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)), 0, 0);
            }
          }
          else
          {
            v44 = *(_QWORD *)(v4 + 24);
            v45 = sub_14C81A0(*(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF)));
            v46 = v44;
            if ( (unsigned __int8)sub_13848E0(v44, v20, 0, v45) )
            {
              v54 = *(_QWORD *)(v4 + 24);
              v55 = sub_14C8160(v46, v20, v47);
              sub_13848E0(v54, v20, 1u, v55);
            }
          }
          if ( v5 != v20 )
            sub_1391C50(v4, v20, v5, 0);
          v19 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
        }
        v16 = 0;
        v17 = v5;
        v18 = *(_QWORD *)(v5 + 24 * (1 - v19));
        return (__int64)sub_13911E0(v4, v18, v17, v16);
      case 0x20:
        return sub_1392250(v4, v5);
      case 0x24:
      case 0x25:
      case 0x26:
      case 0x27:
      case 0x28:
      case 0x29:
      case 0x2A:
      case 0x2B:
      case 0x2C:
      case 0x2F:
      case 0x30:
        v15 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
        result = *(_QWORD *)v15;
        if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 15 )
          goto LABEL_13;
        return result;
      case 0x2D:
        v21 = sub_14C8190();
        v5 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
        v22 = *(_BYTE *)(v5 + 16);
        if ( v22 <= 3u )
          goto LABEL_21;
        if ( v22 == 5 )
        {
          result = (unsigned int)*(unsigned __int16 *)(v5 + 18) - 51;
          if ( (unsigned int)result <= 1 )
            return result;
          v3 &= 0xFFFFFFFF00000000LL;
          a1 = *(_QWORD *)(v4 + 24);
          a2 = v5;
          result = sub_13848E0(a1, v5, v3, 0);
          if ( !(_BYTE)result )
            return result;
          continue;
        }
        v42 = *(_QWORD *)(v4 + 24);
        return sub_13848E0(v42, v5, 0, v21);
      case 0x2E:
        v35 = sub_14C8160(a1, a2, a3);
        v36 = *(_BYTE *)(v5 + 16);
        if ( v36 <= 3u )
        {
LABEL_21:
          v23 = *(_QWORD *)(v4 + 24);
          v24 = sub_14C81A0(v5);
          v25 = v23;
          result = sub_13848E0(v23, v5, 0, v24);
          if ( (_BYTE)result )
          {
            v27 = *(_QWORD *)(v4 + 24);
            v28 = sub_14C8160(v25, v5, v26);
            return sub_13848E0(v27, v5, 1u, v28);
          }
          return result;
        }
        if ( v36 != 5 )
        {
          v42 = *(_QWORD *)(v4 + 24);
          v21 = v35;
          return sub_13848E0(v42, v5, 0, v21);
        }
        result = (unsigned int)*(unsigned __int16 *)(v5 + 18) - 51;
        if ( (unsigned int)result <= 1 )
          return result;
        v56 &= 0xFFFFFFFF00000000LL;
        a2 = v5;
        a1 = *(_QWORD *)(v4 + 24);
        result = sub_13848E0(a1, v5, v56, 0);
        if ( !(_BYTE)result )
          return result;
        continue;
      case 0x37:
        v6 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
        v29 = *(_QWORD *)(v5 + 24 * (1 - v6));
        if ( *(_BYTE *)(*(_QWORD *)v29 + 8LL) != 15 )
          goto LABEL_31;
        result = *(_QWORD *)v5;
        if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 15 )
        {
          v30 = *(_BYTE *)(v29 + 16);
          if ( v30 > 3u )
          {
            if ( v30 == 5 )
            {
              if ( (unsigned int)*(unsigned __int16 *)(v29 + 18) - 51 > 1
                && (unsigned __int8)sub_13848E0(*(_QWORD *)(v4 + 24), v29, 0, 0) )
              {
                sub_1391610(v4, v29);
              }
            }
            else
            {
              sub_13848E0(*(_QWORD *)(v4 + 24), v29, 0, 0);
            }
          }
          else
          {
            v31 = *(_QWORD *)(v4 + 24);
            v32 = sub_14C81A0(v29);
            v33 = v31;
            if ( (unsigned __int8)sub_13848E0(v31, v29, 0, v32) )
            {
              v52 = *(_QWORD *)(v4 + 24);
              v53 = sub_14C8160(v33, v29, v34);
              sub_13848E0(v52, v29, 1u, v53);
            }
          }
          if ( v5 != v29 )
            sub_1391C50(v4, v29, v5, 0);
          v6 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
LABEL_31:
          v14 = 2;
LABEL_32:
          v15 = *(_QWORD *)(v5 + 24 * (v14 - v6));
          result = *(_QWORD *)v15;
          if ( *(_BYTE *)(*(_QWORD *)v15 + 8LL) == 15 )
          {
LABEL_13:
            result = *(_QWORD *)v5;
            if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 15 )
            {
              v37 = *(_BYTE *)(v15 + 16);
              if ( v37 > 3u )
              {
                if ( v37 == 5 )
                {
                  result = (unsigned int)*(unsigned __int16 *)(v15 + 18) - 51;
                  if ( (unsigned int)result > 1 )
                  {
                    result = sub_13848E0(*(_QWORD *)(v4 + 24), v15, 0, 0);
                    if ( (_BYTE)result )
                      result = sub_1391610(v4, v15);
                  }
                }
                else
                {
                  result = sub_13848E0(*(_QWORD *)(v4 + 24), v15, 0, 0);
                }
              }
              else
              {
                v38 = *(_QWORD *)(v4 + 24);
                v39 = sub_14C81A0(v15);
                v40 = v38;
                result = sub_13848E0(v38, v15, 0, v39);
                if ( (_BYTE)result )
                {
                  v48 = *(_QWORD *)(v4 + 24);
                  v49 = sub_14C8160(v40, v15, v41);
                  result = sub_13848E0(v48, v15, 1u, v49);
                }
              }
              if ( v5 != v15 )
                return sub_1391C50(v4, v15, v5, 0);
            }
          }
        }
        return result;
      case 0x3B:
      case 0x3E:
        v16 = 1;
        v17 = v5;
        v18 = *(_QWORD *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
        return (__int64)sub_13911E0(v4, v18, v17, v16);
    }
  }
}
