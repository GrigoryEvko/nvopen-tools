// Function: sub_2255E40
// Address: 0x2255e40
//
_QWORD *__fastcall sub_2255E40(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  bool v3; // zf
  __int64 v4; // rax
  __int64 v5; // rbp
  __int64 v6; // rbp
  __int64 v7; // rbp
  __int64 v8; // rbp
  __int64 v9; // rbp
  __int64 v10; // rbp
  __int64 v11; // rbp
  __int64 v12; // rbp
  __int64 v13; // rbp
  __int64 v14; // rbp
  __int64 v15; // rbp
  __int64 v16; // rbp
  __int64 v17; // rbp
  __int64 v18; // rbp
  __int64 v19; // rbp
  __int64 v20; // rbp
  __int64 v21; // rbp
  __int64 v22; // rbp
  __int64 v23; // rbp
  __int64 v24; // rbp
  __int64 v25; // rbp
  __int64 v26; // rbp
  __int64 v27; // rbp
  __int64 v28; // rbp
  __int64 v29; // rbp
  __int64 v30; // rbp
  __int64 v31; // rbp
  __int64 v32; // rbp
  __int64 v33; // rbp
  __int64 v34; // rbp
  __int64 v35; // rbp
  __int64 v36; // rbp
  __int64 v37; // rbp
  __int64 v38; // rbp
  __int64 v39; // rbp
  __int64 v40; // rbp
  __int64 v41; // rbp
  __int64 v42; // rbp
  __int64 v43; // rbp
  __int64 v44; // rbp
  __int64 v45; // rbp
  __int64 v46; // rbp
  __int64 v47; // rbp
  __int64 v48; // rbp
  __int64 v49; // rbp
  __int64 v50; // rbp
  __int64 v51; // rax
  __int64 v52; // rbx
  _QWORD *result; // rax
  __int64 v54; // rax
  _QWORD v55[4]; // [rsp+8h] [rbp-20h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 16) == 0;
  v55[0] = a2;
  if ( v3 )
  {
    a1 = 400;
    v54 = sub_22077B0(0x190u);
    *(_DWORD *)(v54 + 8) = 0;
    *(_QWORD *)(v54 + 16) = 0;
    *(_QWORD *)v54 = off_4A06C98;
    *(_QWORD *)(v54 + 24) = 0;
    *(_QWORD *)(v54 + 32) = 0;
    *(_QWORD *)(v54 + 40) = 0;
    *(_QWORD *)(v54 + 48) = 0;
    *(_QWORD *)(v54 + 56) = 0;
    *(_QWORD *)(v54 + 64) = 0;
    *(_QWORD *)(v54 + 72) = 0;
    *(_QWORD *)(v54 + 80) = 0;
    *(_QWORD *)(v54 + 88) = 0;
    *(_QWORD *)(v54 + 96) = 0;
    *(_QWORD *)(v54 + 104) = 0;
    *(_QWORD *)(v54 + 112) = 0;
    *(_QWORD *)(v54 + 120) = 0;
    *(_QWORD *)(v54 + 128) = 0;
    *(_QWORD *)(v54 + 136) = 0;
    *(_QWORD *)(v54 + 144) = 0;
    *(_QWORD *)(v54 + 152) = 0;
    *(_QWORD *)(v54 + 160) = 0;
    *(_QWORD *)(v54 + 168) = 0;
    *(_QWORD *)(v54 + 176) = 0;
    *(_QWORD *)(v54 + 184) = 0;
    *(_QWORD *)(v54 + 192) = 0;
    *(_QWORD *)(v54 + 200) = 0;
    *(_QWORD *)(v54 + 208) = 0;
    *(_QWORD *)(v54 + 216) = 0;
    *(_QWORD *)(v54 + 224) = 0;
    *(_QWORD *)(v54 + 232) = 0;
    *(_QWORD *)(v54 + 240) = 0;
    *(_QWORD *)(v54 + 248) = 0;
    *(_QWORD *)(v54 + 256) = 0;
    *(_QWORD *)(v54 + 264) = 0;
    *(_QWORD *)(v54 + 272) = 0;
    *(_QWORD *)(v54 + 280) = 0;
    *(_QWORD *)(v54 + 288) = 0;
    *(_QWORD *)(v54 + 296) = 0;
    *(_QWORD *)(v54 + 304) = 0;
    *(_QWORD *)(v54 + 312) = 0;
    *(_QWORD *)(v54 + 320) = 0;
    *(_QWORD *)(v54 + 328) = 0;
    *(_QWORD *)(v54 + 336) = 0;
    *(_QWORD *)(v54 + 344) = 0;
    *(_QWORD *)(v54 + 352) = 0;
    *(_QWORD *)(v54 + 360) = 0;
    *(_QWORD *)(v54 + 368) = 0;
    *(_QWORD *)(v54 + 376) = 0;
    *(_QWORD *)(v54 + 384) = 0;
    *(_BYTE *)(v54 + 392) = 0;
    *(_QWORD *)(v2 + 16) = v54;
  }
  if ( v55[0] )
  {
    v4 = sub_22542A0(v55);
    v5 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v2 + 24) = v4;
    *(_QWORD *)(v5 + 16) = __nl_langinfo_l();
    v6 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v6 + 24) = __nl_langinfo_l();
    v7 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v7 + 32) = __nl_langinfo_l();
    v8 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v8 + 40) = __nl_langinfo_l();
    v9 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v9 + 48) = __nl_langinfo_l();
    v10 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v10 + 56) = __nl_langinfo_l();
    v11 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v11 + 64) = __nl_langinfo_l();
    v12 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v12 + 72) = __nl_langinfo_l();
    v13 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v13 + 80) = __nl_langinfo_l();
    v14 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v14 + 88) = __nl_langinfo_l();
    v15 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v15 + 96) = __nl_langinfo_l();
    v16 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v16 + 104) = __nl_langinfo_l();
    v17 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v17 + 112) = __nl_langinfo_l();
    v18 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v18 + 120) = __nl_langinfo_l();
    v19 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v19 + 128) = __nl_langinfo_l();
    v20 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v20 + 136) = __nl_langinfo_l();
    v21 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v21 + 144) = __nl_langinfo_l();
    v22 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v22 + 152) = __nl_langinfo_l();
    v23 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v23 + 160) = __nl_langinfo_l();
    v24 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v24 + 168) = __nl_langinfo_l();
    v25 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v25 + 176) = __nl_langinfo_l();
    v26 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v26 + 184) = __nl_langinfo_l();
    v27 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v27 + 192) = __nl_langinfo_l();
    v28 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v28 + 200) = __nl_langinfo_l();
    v29 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v29 + 208) = __nl_langinfo_l();
    v30 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v30 + 216) = __nl_langinfo_l();
    v31 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v31 + 224) = __nl_langinfo_l();
    v32 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v32 + 232) = __nl_langinfo_l();
    v33 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v33 + 240) = __nl_langinfo_l();
    v34 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v34 + 248) = __nl_langinfo_l();
    v35 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v35 + 256) = __nl_langinfo_l();
    v36 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v36 + 264) = __nl_langinfo_l();
    v37 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v37 + 272) = __nl_langinfo_l();
    v38 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v38 + 280) = __nl_langinfo_l();
    v39 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v39 + 288) = __nl_langinfo_l();
    v40 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v40 + 296) = __nl_langinfo_l();
    v41 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v41 + 304) = __nl_langinfo_l();
    v42 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v42 + 312) = __nl_langinfo_l();
    v43 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v43 + 320) = __nl_langinfo_l();
    v44 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v44 + 328) = __nl_langinfo_l();
    v45 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v45 + 336) = __nl_langinfo_l();
    v46 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v46 + 344) = __nl_langinfo_l();
    v47 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v47 + 352) = __nl_langinfo_l();
    v48 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v48 + 360) = __nl_langinfo_l();
    v49 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v49 + 368) = __nl_langinfo_l();
    v50 = *(_QWORD *)(v2 + 16);
    v51 = __nl_langinfo_l();
    v52 = *(_QWORD *)(v2 + 16);
    *(_QWORD *)(v50 + 376) = v51;
    result = (_QWORD *)__nl_langinfo_l();
    *(_QWORD *)(v52 + 384) = result;
  }
  else
  {
    *(_QWORD *)(v2 + 24) = sub_2208E60(a1, a2);
    result = *(_QWORD **)(v2 + 16);
    result[8] = "AM";
    result[9] = "PM";
    result[11] = "Sunday";
    result[12] = "Monday";
    result[13] = "Tuesday";
    result[14] = "Wednesday";
    result[15] = "Thursday";
    result[2] = "%m/%d/%y";
    result[3] = "%m/%d/%y";
    result[16] = "Friday";
    result[4] = "%H:%M:%S";
    result[5] = "%H:%M:%S";
    result[18] = "Sun";
    result[6] = byte_3F871B3;
    result[7] = byte_3F871B3;
    result[10] = byte_3F871B3;
    result[20] = "Tue";
    result[17] = "Saturday";
    result[22] = "Thu";
    result[19] = "Mon";
    result[24] = "Sat";
    result[21] = "Wed";
    result[26] = "February";
    result[23] = "Fri";
    result[28] = "April";
    result[25] = "January";
    result[30] = "June";
    result[27] = "March";
    result[31] = "July";
    result[29] = "May";
    result[32] = "August";
    result[33] = "September";
    result[41] = "May";
    result[34] = "October";
    result[42] = "Jun";
    result[35] = "November";
    result[43] = "Jul";
    result[36] = "December";
    result[44] = "Aug";
    result[37] = "Jan";
    result[45] = "Sep";
    result[38] = "Feb";
    result[46] = "Oct";
    result[39] = "Mar";
    result[47] = "Nov";
    result[40] = "Apr";
    result[48] = "Dec";
  }
  return result;
}
