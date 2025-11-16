// Function: sub_26C2500
// Address: 0x26c2500
//
_BYTE *__fastcall sub_26C2500(_QWORD *a1, __int64 a2)
{
  _BYTE *result; // rax
  unsigned __int32 v5; // r14d
  __int64 v6; // r15
  unsigned __int32 v7; // edx
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  int v22; // r15d
  _BYTE *v23; // rax
  unsigned __int8 v24; // dl
  _BYTE **v25; // rax
  __int64 v26; // rdx
  const char *v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int32 v30; // ebx
  __int64 v31; // r13
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  int v46; // ebx
  _BYTE *v47; // rax
  unsigned __int8 v48; // dl
  _BYTE **v49; // rax
  __int64 v50; // rdx
  const char *v51; // rdi
  unsigned __int8 v52; // dl
  const char **v53; // rax
  unsigned __int8 v54; // dl
  const char **v55; // rax
  __int32 v56; // [rsp+Ch] [rbp-284h]
  __int64 v57; // [rsp+10h] [rbp-280h] BYREF
  __int64 v58; // [rsp+18h] [rbp-278h] BYREF
  __m128i v59[2]; // [rsp+20h] [rbp-270h] BYREF
  __int16 v60; // [rsp+40h] [rbp-250h]
  __m128i v61[2]; // [rsp+50h] [rbp-240h] BYREF
  char v62; // [rsp+70h] [rbp-220h]
  char v63; // [rsp+71h] [rbp-21Fh]
  __m128i v64[3]; // [rsp+80h] [rbp-210h] BYREF
  __m128i v65[2]; // [rsp+B0h] [rbp-1E0h] BYREF
  __int16 v66; // [rsp+D0h] [rbp-1C0h]
  __m128i v67[3]; // [rsp+E0h] [rbp-1B0h] BYREF
  __m128i v68[2]; // [rsp+110h] [rbp-180h] BYREF
  char v69; // [rsp+130h] [rbp-160h]
  char v70; // [rsp+131h] [rbp-15Fh]
  __m128i v71[3]; // [rsp+140h] [rbp-150h] BYREF
  __m128i v72[2]; // [rsp+170h] [rbp-120h] BYREF
  __int16 v73; // [rsp+190h] [rbp-100h]
  __m128i v74[3]; // [rsp+1A0h] [rbp-F0h] BYREF
  __m128i v75[2]; // [rsp+1D0h] [rbp-C0h] BYREF
  char v76; // [rsp+1F0h] [rbp-A0h]
  char v77; // [rsp+1F1h] [rbp-9Fh]
  __m128i v78[3]; // [rsp+200h] [rbp-90h] BYREF
  void *v79; // [rsp+230h] [rbp-60h] BYREF
  __int64 v80; // [rsp+238h] [rbp-58h]
  const char *v81; // [rsp+240h] [rbp-50h]
  __int64 v82; // [rsp+248h] [rbp-48h]
  int v83; // [rsp+250h] [rbp-40h]
  __m128i *v84; // [rsp+258h] [rbp-38h]

  result = &unk_500BC80;
  if ( unk_500BD08 )
  {
    v5 = sub_2A60F00(a1 + 136, a1[150], a1[160]);
    v6 = (unsigned int)sub_2A61050(a1 + 136, a1[150], a1[160]);
    v7 = sub_2A611E0(a1 + 136, v5, v6);
    result = &unk_500BC80;
    if ( v7 < unk_500BD08 )
    {
      v56 = v7;
      v59[0].m128i_i32[0] = v5;
      v8 = sub_B2BE50(a2);
      v75[0].m128i_i64[0] = (__int64)"%) were applied";
      v73 = 265;
      v72[0].m128i_i32[0] = v56;
      v68[0].m128i_i64[0] = (__int64)" available profile records (";
      v65[0].m128i_i32[0] = v6;
      v66 = 265;
      v60 = 265;
      v61[0].m128i_i64[0] = (__int64)" of ";
      v77 = 1;
      v76 = 3;
      v70 = 1;
      v69 = 3;
      v63 = 1;
      v62 = 3;
      sub_9C6370(v64, v59, v61, v9, 265, 265);
      sub_9C6370(v67, v64, v65, v10, v11, v12);
      sub_9C6370(v71, v67, v68, v13, v14, v15);
      sub_9C6370(v74, v71, v72, v16, v17, v18);
      sub_9C6370(v78, v74, v75, v19, v20, v21);
      v22 = sub_26BDAB0(a2);
      v23 = (_BYTE *)sub_B92180(a2);
      if ( *v23 == 16
        || ((v24 = *(v23 - 16), (v24 & 2) != 0)
          ? (v25 = (_BYTE **)*((_QWORD *)v23 - 4))
          : (v25 = (_BYTE **)&v23[-8 * ((v24 >> 2) & 0xF) - 16]),
            (v23 = *v25) != 0) )
      {
        v54 = *(v23 - 16);
        if ( (v54 & 2) != 0 )
          v55 = (const char **)*((_QWORD *)v23 - 4);
        else
          v55 = (const char **)&v23[-8 * ((v54 >> 2) & 0xF) - 16];
        v27 = *v55;
        if ( *v55 )
          v27 = (const char *)sub_B91420((__int64)v27);
        else
          v26 = 0;
      }
      else
      {
        v26 = 0;
        v27 = byte_3F871B3;
      }
      v81 = v27;
      v80 = 0x10000000CLL;
      v82 = v26;
      v83 = v22;
      v79 = &unk_49D9C78;
      v84 = v78;
      result = sub_B6EB20(v8, (__int64)&v79);
    }
  }
  if ( unk_500BC28 )
  {
    v28 = a1[160];
    v29 = a1[150];
    v57 = a1[140];
    v58 = sub_2A61100(a1 + 136, v29, v28);
    result = (_BYTE *)sub_2A611E0(a1 + 136, (unsigned int)v57, v58);
    v30 = (int)result;
    if ( (unsigned int)result < unk_500BC28 )
    {
      v31 = sub_B2BE50(a2);
      v66 = 267;
      v75[0].m128i_i64[0] = (__int64)"%) were applied";
      v73 = 265;
      v68[0].m128i_i64[0] = (__int64)" available profile samples (";
      v65[0].m128i_i64[0] = (__int64)&v58;
      v61[0].m128i_i64[0] = (__int64)" of ";
      v60 = 267;
      v59[0].m128i_i64[0] = (__int64)&v57;
      v72[0].m128i_i32[0] = v30;
      v77 = 1;
      v76 = 3;
      v70 = 1;
      v69 = 3;
      v63 = 1;
      v62 = 3;
      sub_9C6370(v64, v59, v61, 267, v32, v33);
      sub_9C6370(v67, v64, v65, v34, v35, v36);
      sub_9C6370(v71, v67, v68, v37, v38, v39);
      sub_9C6370(v74, v71, v72, v40, v41, v42);
      sub_9C6370(v78, v74, v75, v43, v44, v45);
      v46 = sub_26BDAB0(a2);
      v47 = (_BYTE *)sub_B92180(a2);
      if ( *v47 == 16
        || ((v48 = *(v47 - 16), (v48 & 2) != 0)
          ? (v49 = (_BYTE **)*((_QWORD *)v47 - 4))
          : (v49 = (_BYTE **)&v47[-8 * ((v48 >> 2) & 0xF) - 16]),
            (v47 = *v49) != 0) )
      {
        v52 = *(v47 - 16);
        if ( (v52 & 2) != 0 )
          v53 = (const char **)*((_QWORD *)v47 - 4);
        else
          v53 = (const char **)&v47[-8 * ((v52 >> 2) & 0xF) - 16];
        v51 = *v53;
        if ( *v53 )
          v51 = (const char *)sub_B91420((__int64)v51);
        else
          v50 = 0;
      }
      else
      {
        v50 = 0;
        v51 = byte_3F871B3;
      }
      v81 = v51;
      v80 = 0x10000000CLL;
      v82 = v50;
      v83 = v46;
      v79 = &unk_49D9C78;
      v84 = v78;
      return sub_B6EB20(v31, (__int64)&v79);
    }
  }
  return result;
}
