// Function: sub_5EE990
// Address: 0x5ee990
//
__int64 *__fastcall sub_5EE990(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        _QWORD *a4,
        int a5,
        __int64 a6,
        __int64 *a7,
        char a8,
        __m128i *a9,
        int a10)
{
  __int64 v11; // r12
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 *result; // rax
  __int64 v16; // rcx
  __int64 v17; // rax
  _QWORD *i; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // r8
  char v23; // al
  __int64 v24; // r13
  __int64 v25; // r14
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // r8
  char v29; // al
  __int64 v30; // rbx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 j; // rsi
  __int64 v34; // rdx
  __int64 v35; // r9
  __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // [rsp+8h] [rbp-68h]
  __int64 v40; // [rsp+8h] [rbp-68h]
  _QWORD *v42; // [rsp+10h] [rbp-60h]
  __int64 v43; // [rsp+10h] [rbp-60h]
  __int64 v44; // [rsp+10h] [rbp-60h]
  __int64 v45; // [rsp+10h] [rbp-60h]
  _QWORD v48[7]; // [rsp+38h] [rbp-38h] BYREF

  v11 = a1;
  if ( *(_BYTE *)(a1 + 80) == 16 )
    v11 = **(_QWORD **)(a1 + 88);
  v48[0] = unk_4D04A08;
  if ( !(unsigned int)sub_884000(a1, 0) )
    return (__int64 *)sub_6854C0(265, v48, a1);
  if ( *(_BYTE *)(a1 + 80) != 10
    || (result = *(__int64 **)(a1 + 88), (*((_BYTE *)result + 193) & 0x10) == 0)
    || *((_BYTE *)result + 174) == 5 && *((_BYTE *)result + 176) == 15 )
  {
    if ( !*a3 || (result = (__int64 *)sub_5EE800(v11, *a3, (__int64)v48, v13, v14), !(_DWORD)result) )
    {
      if ( !a5 && a2 != v11 )
      {
        v16 = *(_QWORD *)(v11 + 64);
        v17 = *(_QWORD *)(a2 + 64);
        if ( v16 != v17 )
        {
          if ( !v16 || !v17 || !dword_4F07588 || (v37 = *(_QWORD *)(v16 + 32), *(_QWORD *)(v17 + 32) != v37) || !v37 )
          {
            for ( i = **(_QWORD ***)(a6 + 168); ; i = (_QWORD *)*i )
            {
              v20 = i[5];
              if ( v20 != v16 )
              {
                if ( !v20 )
                  continue;
                if ( !v16 )
                  continue;
                if ( !dword_4F07588 )
                  continue;
                v19 = *(_QWORD *)(v20 + 32);
                if ( *(_QWORD *)(v16 + 32) != v19 || !v19 )
                  continue;
              }
              v42 = i;
              v21 = sub_8D5D50(i, a4);
              i = v42;
              if ( v21 )
              {
                a4 = v42;
                break;
              }
              v16 = *(_QWORD *)(v11 + 64);
            }
          }
        }
      }
      v22 = sub_87F190(a1, a6, a4, 0, 0);
      v23 = *(_BYTE *)(v22 + 96) & 0xF8 | a8 & 3 | 4;
      *(_BYTE *)(v22 + 96) = v23;
      if ( *(_BYTE *)(v11 + 80) == 2 )
      {
        v36 = *(_QWORD *)(v11 + 88);
        if ( v36 )
        {
          if ( *(_BYTE *)(v36 + 173) == 12 )
            *(_BYTE *)(v22 + 96) = v23 | 0x10;
        }
      }
      *(_QWORD *)(*(_QWORD *)(v22 + 88) + 16LL) = unk_4D04A20;
      *(_QWORD *)(v22 + 48) = v48[0];
      if ( *a3 )
      {
        v39 = *a3;
        v24 = *(_QWORD *)(*(_QWORD *)a6 + 96LL);
        v43 = v22;
        v25 = *(_QWORD *)(v24 + 32);
        v26 = sub_887160(v22, *a3, 0, 0);
        *a3 = v26;
        if ( v39 == v25 )
          *(_QWORD *)(v24 + 32) = v26;
        sub_5EE4B0(v26);
        v28 = v43;
      }
      else
      {
        v35 = *(_QWORD *)(*(_QWORD *)v22 + 24LL);
        if ( v35
          && *(_DWORD *)(v22 + 40) == *(_DWORD *)(v35 + 40)
          && *(_BYTE *)(v11 + 80) == 2
          && (v38 = *(_QWORD *)(v11 + 88)) != 0
          && *(_BYTE *)(v38 + 173) == 12 )
        {
          v40 = *(_QWORD *)(*(_QWORD *)v22 + 24LL);
          v45 = v22;
          sub_881DB0(v40);
          sub_885FF0(v45, (unsigned int)dword_4F04C64, 1);
          if ( (*(_BYTE *)(v45 + 81) & 0x20) == 0 )
            *a3 = v45;
          sub_885FF0(v40, (unsigned int)dword_4F04C64, 1);
          v28 = v45;
        }
        else
        {
          v44 = v22;
          sub_885FF0(v22, (unsigned int)dword_4F04C64, 1);
          v28 = v44;
          if ( (*(_BYTE *)(v44 + 81) & 0x20) == 0 )
          {
            *a3 = v44;
            v29 = *(_BYTE *)(v11 + 80);
            if ( v29 != 10 )
              goto LABEL_33;
            goto LABEL_61;
          }
        }
      }
      v29 = *(_BYTE *)(v11 + 80);
      if ( v29 != 10 )
      {
LABEL_33:
        if ( v29 != 20 )
          goto LABEL_34;
        v32 = *(_QWORD *)(*(_QWORD *)(v11 + 88) + 176LL);
LABEL_45:
        if ( !v32 )
        {
LABEL_38:
          v30 = sub_6506C0(v11, v48, (unsigned int)dword_4F04C64, v27, v28);
          v31 = *(_QWORD *)(a2 + 64);
          *(_BYTE *)(v30 + 40) |= 2u;
          *(_QWORD *)(v30 + 48) = v31;
          *(_BYTE *)(v30 + 42) = a8;
          if ( a9 )
          {
            if ( a10 )
              a9 = (__m128i *)sub_5CF190(a9);
            sub_5CEC90(a9, v30, 29);
          }
          sub_876960(v11, v48, v30, *a7);
          *a7 = v30;
          return a7;
        }
        v27 = *(unsigned __int8 *)(v32 + 174);
        if ( (_BYTE)v27 == 3 )
        {
          while ( *(_BYTE *)(a6 + 140) == 12 )
            a6 = *(_QWORD *)(a6 + 160);
          sub_5E6C40(v28, *(_QWORD *)(*(_QWORD *)a6 + 96LL));
          v29 = *(_BYTE *)(v11 + 80);
        }
        else if ( (_BYTE)v27 == 5 && *(_BYTE *)(v32 + 176) == 15 )
        {
          for ( j = *a3; *(_BYTE *)(a6 + 140) == 12; a6 = *(_QWORD *)(a6 + 160) )
            ;
          v34 = *(_QWORD *)(*(_QWORD *)a6 + 96LL);
          v27 = *(_QWORD *)(v34 + 32);
          if ( v27 )
          {
            if ( *(_BYTE *)(v27 + 80) != 17 )
            {
              *(_QWORD *)(v34 + 32) = j;
              v29 = *(_BYTE *)(v11 + 80);
            }
          }
          else
          {
            if ( !j )
              j = v28;
            *(_QWORD *)(v34 + 32) = j;
            v29 = *(_BYTE *)(v11 + 80);
          }
        }
LABEL_34:
        if ( v29 == 16 )
        {
          v11 = **(_QWORD **)(v11 + 88);
          v29 = *(_BYTE *)(v11 + 80);
        }
        if ( v29 == 24 )
          v11 = *(_QWORD *)(v11 + 88);
        goto LABEL_38;
      }
LABEL_61:
      v32 = *(_QWORD *)(v11 + 88);
      goto LABEL_45;
    }
  }
  return result;
}
